from __future__ import annotations

import pytest

from jimmy_assistant.nlp import intent as A
from jimmy_assistant.nlp.rules import RulesParser


parser = RulesParser()


class TestPower:
    @pytest.mark.parametrize(
        "utterance",
        [
            "hibernate",
            "hibernate the pc",
            "hibernate the computer",
            "pc ko hibernate",
            "system hibernate kar do",
        ],
    )
    def test_hibernate(self, utterance: str) -> None:
        intent = parser.parse(utterance)
        assert intent is not None and intent.name == A.ACTION_HIBERNATE

    @pytest.mark.parametrize(
        "utterance",
        [
            "shutdown",
            "shut down the pc",
            "power off",
            "switch off the laptop",
            "turn off the computer",
            "pc band kar do",
            "computer bandh karo",
            "laptop ko bandh kar do",
        ],
    )
    def test_shutdown(self, utterance: str) -> None:
        intent = parser.parse(utterance)
        assert intent is not None and intent.name == A.ACTION_SHUTDOWN

    @pytest.mark.parametrize(
        "utterance",
        [
            "restart",
            "reboot",
            "restart the pc",
            "pc restart kar do",
            "reboot the system",
        ],
    )
    def test_restart(self, utterance: str) -> None:
        intent = parser.parse(utterance)
        assert intent is not None and intent.name == A.ACTION_RESTART

    @pytest.mark.parametrize(
        "utterance",
        [
            "sleep",
            "go to sleep",
            "put the pc to sleep",
            "pc ko sula do",
        ],
    )
    def test_sleep(self, utterance: str) -> None:
        intent = parser.parse(utterance)
        assert intent is not None and intent.name == A.ACTION_SLEEP

    @pytest.mark.parametrize(
        "utterance",
        [
            "lock",
            "lock the pc",
            "lock the screen",
            "lock the computer",
            "pc lock kar do",
            "screen lock karo",
        ],
    )
    def test_lock(self, utterance: str) -> None:
        intent = parser.parse(utterance)
        assert intent is not None and intent.name == A.ACTION_LOCK


class TestVolume:
    @pytest.mark.parametrize(
        "utterance",
        ["mute", "volume band karo", "awaaz mute kar do"],
    )
    def test_mute(self, utterance: str) -> None:
        assert parser.parse(utterance).name == A.ACTION_VOLUME_MUTE

    @pytest.mark.parametrize(
        "utterance",
        ["unmute", "volume chalu kar do", "awaaz chalao"],
    )
    def test_unmute(self, utterance: str) -> None:
        assert parser.parse(utterance).name == A.ACTION_VOLUME_UNMUTE

    @pytest.mark.parametrize(
        "utterance, level",
        [
            ("set volume to 50", "50"),
            ("volume 30 percent", "30"),
            ("volume to 100", "100"),
            ("volume 70 pe kar do", "70"),
        ],
    )
    def test_set(self, utterance: str, level: str) -> None:
        intent = parser.parse(utterance)
        assert intent is not None
        assert intent.name == A.ACTION_VOLUME_SET
        assert intent.slots.get("level") == level

    def test_up(self) -> None:
        assert parser.parse("volume up").name == A.ACTION_VOLUME_UP
        assert parser.parse("increase the volume").name == A.ACTION_VOLUME_UP
        assert parser.parse("awaaz badha do").name == A.ACTION_VOLUME_UP

    def test_down(self) -> None:
        assert parser.parse("volume down").name == A.ACTION_VOLUME_DOWN
        assert parser.parse("awaaz kam kar do").name == A.ACTION_VOLUME_DOWN


class TestMedia:
    @pytest.mark.parametrize("utt", ["play", "pause", "resume", "pause the music", "gaana rok do"])
    def test_play_pause(self, utt: str) -> None:
        # Note: "play" alone falls through youtube.play in the pattern list —
        # but with no query it should be treated as media toggle. Actually
        # with our current rules, bare "play" matches media.play_pause first.
        intent = parser.parse(utt)
        assert intent is not None
        assert intent.name == A.ACTION_MEDIA_PLAY_PAUSE

    def test_next(self) -> None:
        assert parser.parse("next song").name == A.ACTION_MEDIA_NEXT
        assert parser.parse("skip track").name == A.ACTION_MEDIA_NEXT
        assert parser.parse("agla gaana").name == A.ACTION_MEDIA_NEXT

    def test_previous(self) -> None:
        assert parser.parse("previous song").name == A.ACTION_MEDIA_PREV
        assert parser.parse("pichla gaana").name == A.ACTION_MEDIA_PREV


class TestYoutube:
    @pytest.mark.parametrize(
        "utterance, expected_query",
        [
            ("play aaoge jab tum on youtube", "aaoge jab tum"),
            ("aaoge jab tum youtube pe chala do", "aaoge jab tum"),
            ("aaoge jab tum ko youtube pe chala do", "aaoge jab tum"),
            ("youtube pe aaoge jab tum chala do", "aaoge jab tum"),
            ("play tum hi ho", "tum hi ho"),
            ("aaoge jab tum chala do", "aaoge jab tum"),
        ],
    )
    def test_play(self, utterance: str, expected_query: str) -> None:
        intent = parser.parse(utterance)
        assert intent is not None
        assert intent.name == A.ACTION_YOUTUBE_PLAY
        assert intent.slots.get("query") == expected_query


class TestApps:
    @pytest.mark.parametrize(
        "utterance, expected_app",
        [
            ("open chrome", "chrome"),
            ("launch notepad", "notepad"),
            ("start file explorer", "file explorer"),
            ("chrome kholo", "chrome"),
            ("notepad khol do", "notepad"),
            ("vs code start karo", "vs code"),
        ],
    )
    def test_launch(self, utterance: str, expected_app: str) -> None:
        intent = parser.parse(utterance)
        assert intent is not None
        assert intent.name == A.ACTION_APP_LAUNCH
        assert intent.slots.get("app") == expected_app

    @pytest.mark.parametrize(
        "utterance, expected_app",
        [
            ("close edge browser", "edge"),
            ("quit chrome", "chrome"),
            ("notepad band karo", "notepad"),
        ],
    )
    def test_close(self, utterance: str, expected_app: str) -> None:
        intent = parser.parse(utterance)
        assert intent is not None
        assert intent.name == A.ACTION_APP_CLOSE
        assert intent.slots.get("app") == expected_app


class TestWhatsapp:
    def test_send_message_pattern(self) -> None:
        intent = parser.parse(
            "open whatsapp and send a message to sid original saying I will be there in 10 minutes"
        )
        assert intent is not None
        assert intent.name == A.ACTION_WHATSAPP_SEND
        assert intent.slots.get("contact") == "sid original"
        assert "10 minutes" in (intent.slots.get("message") or "")


class TestWebSearch:
    def test_search_google(self) -> None:
        intent = parser.parse("google weather in mumbai")
        assert intent is not None
        assert intent.name == A.ACTION_WEB_SEARCH
        assert intent.slots.get("query") == "weather in mumbai"

    def test_search_for(self) -> None:
        intent = parser.parse("search for python tutorials")
        assert intent is not None
        assert intent.name == A.ACTION_WEB_SEARCH


class TestOpen:
    def test_folder(self) -> None:
        intent = parser.parse("open the folder downloads")
        assert intent is not None
        assert intent.name == A.ACTION_OPEN_THING
        assert intent.slots.get("target") == "downloads"

    def test_file(self) -> None:
        intent = parser.parse("open the file notes")
        assert intent is not None
        assert intent.name == A.ACTION_OPEN_THING
        assert intent.slots.get("target") == "notes"


class TestCancelAndFallthrough:
    @pytest.mark.parametrize("utt", ["cancel", "stop", "abort", "nevermind", "never mind", "ruko", "rehne do"])
    def test_cancel(self, utt: str) -> None:
        intent = parser.parse(utt)
        assert intent is not None
        assert intent.name == A.ACTION_CANCEL

    def test_unknown_returns_none(self) -> None:
        assert parser.parse("gibberish nonsense floop") is None
        assert parser.parse("") is None
