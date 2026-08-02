"""Multi-step tool-calling agent.

Given a user utterance, the agent asks the local Ollama model which
tool(s) to call, executes them via the :class:`ActionRegistry`, feeds
the results back into the conversation, and repeats until the model
stops requesting tools (task complete) or a safety limit is hit.

Design notes
------------
* **Rules fast-path** stays as a shortcut layer above the agent — the
  orchestrator in :mod:`jimmy_assistant.main` calls the rules parser
  first, and only falls into the agent loop when rules don't match
  cleanly. This keeps latency low for known commands ("lock the pc")
  while still supporting arbitrary agentic phrasing for the rest.
* **Confirmation** for destructive tools is enforced *by the caller*,
  not by the LLM. When the agent picks a destructive tool the
  orchestrator interrupts, asks the user to say yes/no, and only then
  dispatches the handler. This keeps the safety boundary tight even
  if the model tries to bypass it.
* **Failure isolation**: any tool crash is caught, converted to an
  ``ActionResult.failure``, and fed back to the model as a "tool
  error" message so it can try a different approach or apologise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from jimmy_assistant.actions.registry import ActionRegistry, ActionResult
from jimmy_assistant.nlp.agent_prompt import AGENT_SYSTEM_PROMPT
from jimmy_assistant.nlp.intent import Intent
from jimmy_assistant.nlp.ollama_client import ChatResponse, OllamaClient, ToolCall
from jimmy_assistant.ui.events import JimmyListener, NullListener


logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """One iteration of the agent loop, for logging / debugging."""

    tool_name: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    result_ok: bool = True
    result_summary: str = ""


@dataclass
class AgentOutcome:
    """Final result of an agent run — everything the orchestrator
    needs to speak back and log."""

    steps: List[AgentStep] = field(default_factory=list)
    final_message: str = ""
    reached_limit: bool = False
    used_llm: bool = False

    @property
    def any_action_executed(self) -> bool:
        return any(s.tool_name for s in self.steps)


# ConfirmCallback: given an intent, return True if the user approved.
ConfirmCallback = Callable[[Intent], bool]


class Agent:
    """Drives a tool-calling loop against a local Ollama model.

    Parameters
    ----------
    registry
        The whitelist of executable tools.
    llm
        Ollama client. Must expose ``chat(messages, tools)``.
    confirm_cb
        Optional callable invoked before dispatching any destructive
        tool. Must return True to allow execution, False to cancel.
    max_steps
        Hard cap on tool-call rounds to prevent runaway loops.
    """

    def __init__(
        self,
        registry: ActionRegistry,
        llm: OllamaClient,
        confirm_cb: Optional[ConfirmCallback] = None,
        max_steps: int = 4,
        listener: Optional[JimmyListener] = None,
    ) -> None:
        self._registry = registry
        self._llm = llm
        self._confirm = confirm_cb
        self._max_steps = max_steps
        self._listener: JimmyListener = listener or NullListener()

    # ------------------------------------------------------------------
    def run(self, transcript: str) -> AgentOutcome:
        outcome = AgentOutcome(used_llm=True)
        tools = self._registry.openai_tools()
        if not tools:
            logger.error("Agent has no tools registered; cannot proceed")
            outcome.final_message = "I have no tools available."
            return outcome

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ]
        call_counts: Dict[tuple[str, str], int] = {}

        for step_idx in range(self._max_steps):
            response = self._llm.chat(messages=messages, tools=tools)
            if not response.ok:
                logger.warning("Agent LLM error at step %d: %s", step_idx, response.error)
                outcome.final_message = "Sorry, I couldn't reach the AI right now."
                return outcome

            # If the model gave a natural-language reply and no tool calls,
            # we're done — surface the reply.
            if not response.tool_calls:
                outcome.final_message = response.content or ""
                logger.info("Agent finished with message: %r", outcome.final_message)
                return outcome

            # Otherwise execute each requested tool call.
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    }
                    for tc in response.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for call in response.tool_calls:
                args_json = json.dumps(call.arguments, sort_keys=True, ensure_ascii=True)
                signature = (call.name, args_json)
                seen = call_counts.get(signature, 0)
                call_counts[signature] = seen + 1

                if seen >= 1:
                    step = AgentStep(
                        tool_name=call.name,
                        arguments=dict(call.arguments),
                        result_ok=False,
                        result_summary=(
                            "duplicate tool call blocked; use previous tool result and respond"
                        ),
                    )
                    logger.warning("Blocked duplicate tool call %s args=%s", call.name, args_json)
                    self._listener.on_tool_result(call.name, False, step.result_summary)
                else:
                    step = self._execute_call(call)
                outcome.steps.append(step)
                messages.append(
                    {
                        "role": "tool",
                        "tool_name": call.name,
                        "content": json.dumps(
                            {"ok": step.result_ok, "message": step.result_summary}
                        ),
                    }
                )

            # If every call so far succeeded and the model didn't request
            # anything requiring further planning, we can early-exit
            # after one round to keep latency low. We do that by asking
            # the model for a short summary in the *next* iteration —
            # which it can answer without a tool call, ending the loop.

        outcome.reached_limit = True
        logger.warning("Agent hit max_steps=%d", self._max_steps)
        return outcome

    # ------------------------------------------------------------------
    def _execute_call(self, call: ToolCall) -> AgentStep:
        step = AgentStep(tool_name=call.name, arguments=dict(call.arguments))
        self._listener.on_tool_call(call.name, dict(call.arguments))

        if not self._registry.has(call.name):
            step.result_ok = False
            step.result_summary = f"unknown tool: {call.name}"
            logger.warning("LLM tried unknown tool %r", call.name)
            self._listener.on_tool_result(call.name, False, step.result_summary)
            return step

        intent = self._registry.build_intent(call.name, call.arguments)

        # Enforce verbal confirmation for destructive tools BEFORE dispatch.
        if self._registry.is_destructive(call.name):
            if self._confirm is None or not self._confirm(intent):
                step.result_ok = False
                step.result_summary = "user did not confirm"
                logger.info("Destructive tool %s cancelled by user", call.name)
                self._listener.on_tool_result(call.name, False, step.result_summary)
                return step

        result = self._registry.dispatch(intent)
        step.result_ok = result.ok
        if result.ok:
            step.result_summary = result.speak_en or "done"
        else:
            step.result_summary = f"error: {result.error}"
        self._listener.on_tool_result(call.name, step.result_ok, step.result_summary)
        return step

    # -- convenience --------------------------------------------------
    def final_utterance(self, outcome: AgentOutcome, lang: str = "en") -> str:
        """Pick the best line to speak back to the user."""
        if outcome.final_message:
            return outcome.final_message
        # No LLM reply: prefer the last successful tool before surfacing errors.
        for step in reversed(outcome.steps):
            if step.tool_name and step.result_ok:
                return step.result_summary or "Done."
        for step in reversed(outcome.steps):
            if step.tool_name:
                return f"Sorry, that didn't work: {step.result_summary}"
        return "Sorry, I didn't do anything."


# ---------------------------------------------------------------------------
# Helper — build an :class:`ActionResult`-like reply from an :class:`AgentOutcome`
# ---------------------------------------------------------------------------
def outcome_to_result(outcome: AgentOutcome) -> ActionResult:
    """Adapt an :class:`AgentOutcome` to the older :class:`ActionResult`
    shape so callers that already know how to speak an ``ActionResult``
    can be reused."""
    if not outcome.any_action_executed and outcome.final_message:
        return ActionResult.success(speak_en=outcome.final_message)
    last = outcome.steps[-1] if outcome.steps else None
    if last is None:
        return ActionResult.failure("no action")
    if last.result_ok:
        return ActionResult.success(speak_en=last.result_summary)
    return ActionResult.failure(last.result_summary)
