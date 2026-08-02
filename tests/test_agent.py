from __future__ import annotations

from typing import Any, Dict, List

from jimmy_assistant.actions.registry import ActionRegistry, ActionResult, ToolSchema
from jimmy_assistant.nlp.agent import Agent
from jimmy_assistant.nlp.intent import Intent
from jimmy_assistant.nlp.ollama_client import ChatResponse, ToolCall


class _FakeLLM:
    """Deterministic Ollama stand-in for agent unit tests.

    ``script`` is a list of :class:`ChatResponse` objects that will be
    returned in order, one per :meth:`chat` call. If more calls happen
    than there are scripted responses, an empty response is returned.
    """

    def __init__(self, script: List[ChatResponse]) -> None:
        self._script = list(script)
        self.calls: List[Dict[str, Any]] = []

    def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
    ) -> ChatResponse:
        self.calls.append({"messages": messages, "tools": tools or []})
        if not self._script:
            return ChatResponse(content="")
        return self._script.pop(0)


def _handler_ok(intent: Intent) -> ActionResult:
    return ActionResult.success(speak_en=f"ran {intent.name}")


def _handler_boom(_intent: Intent) -> ActionResult:
    return ActionResult.failure("nope")


def _make_registry() -> ActionRegistry:
    r = ActionRegistry()
    r.register(
        "answer.direct",
        _handler_ok,
        schema=ToolSchema(
            description="Answer directly.",
            parameters={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        ),
    )
    r.register(
        "youtube.play",
        _handler_ok,
        schema=ToolSchema(
            description="Play a video on YouTube.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
    )
    r.register(
        "power.hibernate",
        _handler_ok,
        destructive=True,
        schema=ToolSchema(description="Hibernate the pc.", parameters={"type": "object", "properties": {}, "required": []}),
    )
    r.register(
        "power.fail",
        _handler_boom,
        schema=ToolSchema(description="Broken.", parameters={"type": "object", "properties": {}, "required": []}),
    )
    r.register(
        "window.control",
        _handler_ok,
        schema=ToolSchema(
            description="Control current window.",
            parameters={
                "type": "object",
                "properties": {"operation": {"type": "string"}},
                "required": ["operation"],
            },
        ),
    )
    return r


class TestAgentSingleStep:
    def test_generic_date_question_uses_answer_tool(self) -> None:
        llm = _FakeLLM(
            [
                ChatResponse(
                    tool_calls=[
                        ToolCall(
                            name="answer.direct",
                            arguments={
                                "answer": "From 21 March to 11 July is 112 days, exactly 16 weeks."
                            },
                        )
                    ]
                ),
                ChatResponse(content="From 21 March to 11 July is 112 days, exactly 16 weeks."),
            ]
        )
        agent = Agent(registry=_make_registry(), llm=llm)
        outcome = agent.run("Today is 11th July; how many weeks since 21st March?")

        assert outcome.steps[0].tool_name == "answer.direct"
        assert outcome.steps[0].arguments["answer"].endswith("16 weeks.")
        assert outcome.final_message.endswith("16 weeks.")

    def test_calls_tool_and_returns(self) -> None:
        llm = _FakeLLM(
            [
                ChatResponse(
                    tool_calls=[ToolCall(name="youtube.play", arguments={"query": "aaoge jab tum"})]
                ),
                ChatResponse(content="Playing on YouTube."),
            ]
        )
        agent = Agent(registry=_make_registry(), llm=llm)
        outcome = agent.run("play aaoge jab tum on youtube")

        assert len(outcome.steps) == 1
        assert outcome.steps[0].tool_name == "youtube.play"
        assert outcome.steps[0].arguments == {"query": "aaoge jab tum"}
        assert outcome.steps[0].result_ok is True
        assert outcome.final_message == "Playing on YouTube."
        assert not outcome.reached_limit

    def test_text_only_response(self) -> None:
        llm = _FakeLLM([ChatResponse(content="I can't do that.")])
        agent = Agent(registry=_make_registry(), llm=llm)
        outcome = agent.run("hopeless request")

        assert outcome.steps == []
        assert outcome.final_message == "I can't do that."


class TestAgentSafety:
    def test_destructive_denied(self) -> None:
        llm = _FakeLLM(
            [
                ChatResponse(tool_calls=[ToolCall(name="power.hibernate", arguments={})]),
                ChatResponse(content="ok cancelled"),
            ]
        )
        agent = Agent(
            registry=_make_registry(),
            llm=llm,
            confirm_cb=lambda _intent: False,
        )
        outcome = agent.run("hibernate the pc")

        assert outcome.steps[0].result_ok is False
        assert "did not confirm" in outcome.steps[0].result_summary

    def test_destructive_approved(self) -> None:
        llm = _FakeLLM(
            [
                ChatResponse(tool_calls=[ToolCall(name="power.hibernate", arguments={})]),
                ChatResponse(content="hibernating"),
            ]
        )
        agent = Agent(
            registry=_make_registry(),
            llm=llm,
            confirm_cb=lambda _intent: True,
        )
        outcome = agent.run("hibernate")

        assert outcome.steps[0].result_ok is True
        assert outcome.steps[0].tool_name == "power.hibernate"

    def test_unknown_tool(self) -> None:
        llm = _FakeLLM(
            [
                ChatResponse(tool_calls=[ToolCall(name="nonexistent.tool", arguments={})]),
                ChatResponse(content="oh well"),
            ]
        )
        agent = Agent(registry=_make_registry(), llm=llm)
        outcome = agent.run("do impossible thing")

        assert outcome.steps[0].result_ok is False
        assert "unknown tool" in outcome.steps[0].result_summary

    def test_handler_failure_is_reported(self) -> None:
        llm = _FakeLLM(
            [
                ChatResponse(tool_calls=[ToolCall(name="power.fail", arguments={})]),
                ChatResponse(content="that failed"),
            ]
        )
        agent = Agent(registry=_make_registry(), llm=llm)
        outcome = agent.run("break something")

        assert outcome.steps[0].result_ok is False
        assert "nope" in outcome.steps[0].result_summary


class TestAgentLimits:
    def test_reaches_max_steps(self) -> None:
        # Have the LLM keep asking for tool calls forever.
        script = [
            ChatResponse(tool_calls=[ToolCall(name="youtube.play", arguments={"query": "x"})])
            for _ in range(10)
        ]
        llm = _FakeLLM(script)
        agent = Agent(registry=_make_registry(), llm=llm, max_steps=3)
        outcome = agent.run("loop forever")

        assert outcome.reached_limit is True
        assert len(outcome.steps) >= 3

    def test_blocks_duplicate_tool_calls(self) -> None:
        llm = _FakeLLM(
            [
                ChatResponse(tool_calls=[ToolCall(name="youtube.play", arguments={"query": "x"})]),
                ChatResponse(tool_calls=[ToolCall(name="youtube.play", arguments={"query": "x"})]),
                ChatResponse(content="Done."),
            ]
        )
        agent = Agent(registry=_make_registry(), llm=llm, max_steps=4)
        outcome = agent.run("play x")

        assert len(outcome.steps) == 2
        assert outcome.steps[0].result_ok is True
        assert outcome.steps[1].result_ok is False
        assert "duplicate tool call blocked" in outcome.steps[1].result_summary
        assert outcome.final_message == "Done."


class TestRegistryToolsAPI:
    def test_openai_tools_schema(self) -> None:
        r = _make_registry()
        tools = r.openai_tools()
        names = {t["function"]["name"] for t in tools}
        assert "youtube.play" in names
        assert "power.hibernate" in names
        assert "window.control" in names
        yt = next(t for t in tools if t["function"]["name"] == "youtube.play")
        assert yt["function"]["parameters"]["required"] == ["query"]

    def test_build_intent_from_arguments(self) -> None:
        r = _make_registry()
        intent = r.build_intent("youtube.play", {"query": "aaoge jab tum"})
        assert intent.name == "youtube.play"
        assert intent.slots == {"query": "aaoge jab tum"}
        assert intent.source == "llm"
