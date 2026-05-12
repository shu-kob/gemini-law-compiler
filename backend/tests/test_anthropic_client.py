"""Unit tests for src/llm/anthropic_client.py

`anthropic.AnthropicVertex` を fake クラスに差し替え、リクエスト引数の構築
(thinking config / temperature 抑制 / system / max_tokens) を検証する。
ネットワーク・GCP 認証は触らない。
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.llm.anthropic_client import AnthropicResponse


# ---------------------------------------------------------------------------
# Fake AnthropicVertex 実装
# ---------------------------------------------------------------------------
@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeNonTextBlock:
    """text 以外の block (例: thinking) — 出力テキストには含まれないこと。"""

    type: str = "thinking"


@dataclass
class _FakeMessage:
    content: list


@dataclass
class _FakeMessages:
    captured_kwargs: dict = field(default_factory=dict)
    blocks: list = field(default_factory=list)

    def create(self, **kwargs) -> _FakeMessage:
        self.captured_kwargs.update(kwargs)
        return _FakeMessage(content=self.blocks)


@dataclass
class _FakeAnthropicVertex:
    project_id: str
    region: str
    messages: _FakeMessages = field(default_factory=_FakeMessages)


@pytest.fixture
def install_fake_anthropic(monkeypatch: pytest.MonkeyPatch):
    """`from anthropic import AnthropicVertex` を fake に差し替えるファクトリ。

    blocks を指定すると create() がそれらを content に持たせて返す。
    """

    def _install(blocks: list | None = None) -> _FakeMessages:
        fake_module = types.ModuleType("anthropic")

        captured = _FakeMessages(blocks=blocks if blocks is not None else [_FakeTextBlock("ok")])

        def factory(project_id: str, region: str = "global"):
            instance = _FakeAnthropicVertex(project_id=project_id, region=region)
            instance.messages = captured
            return instance

        fake_module.AnthropicVertex = factory  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)
        return captured

    return _install


# ---------------------------------------------------------------------------
# __init__: 認証エラー / 依存欠落のラップ
# ---------------------------------------------------------------------------
class TestInit:
    def test_missing_project_id_raises(self, install_fake_anthropic) -> None:
        install_fake_anthropic()
        from src.llm.anthropic_client import AnthropicClient

        with pytest.raises(RuntimeError) as ei:
            AnthropicClient(project_id="")
        assert "GOOGLE_CLOUD_PROJECT" in str(ei.value)

    def test_missing_anthropic_dep_is_wrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # `anthropic` パッケージが import できない状況を再現
        fake_module = types.ModuleType("anthropic")
        # AnthropicVertex 属性を持たせない → ImportError 経路へ
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)

        from src.llm.anthropic_client import AnthropicClient

        with pytest.raises(RuntimeError) as ei:
            AnthropicClient(project_id="some-project")
        # ユーザー向けインストール手順が出る
        assert "anthropic[vertex]" in str(ei.value)

    def test_default_region_is_global(self, install_fake_anthropic) -> None:
        install_fake_anthropic()
        from src.llm.anthropic_client import AnthropicClient

        client = AnthropicClient(project_id="my-proj")
        # fake では region がインスタンスに保存される
        assert client._client.region == "global"

    def test_explicit_region_is_used(self, install_fake_anthropic) -> None:
        install_fake_anthropic()
        from src.llm.anthropic_client import AnthropicClient

        client = AnthropicClient(project_id="my-proj", region="us-east5")
        assert client._client.region == "us-east5"


# ---------------------------------------------------------------------------
# generate_content: リクエスト引数の構築
# ---------------------------------------------------------------------------
class TestGenerateContentRequest:
    def test_returns_anthropic_response(self, install_fake_anthropic) -> None:
        install_fake_anthropic([_FakeTextBlock("回答テキスト")])
        from src.llm.anthropic_client import AnthropicClient

        client = AnthropicClient(project_id="p")
        result = client.models.generate_content(
            "claude-sonnet-4-6@default", "質問"
        )
        assert isinstance(result, AnthropicResponse)
        assert result.text == "回答テキスト"

    def test_message_role_is_user(self, install_fake_anthropic) -> None:
        captured = install_fake_anthropic()
        from src.llm.anthropic_client import AnthropicClient

        AnthropicClient(project_id="p").models.generate_content(
            "claude-sonnet-4-6@default", "Q"
        )
        messages = captured.captured_kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Q"

    def test_default_max_tokens_is_set(self, install_fake_anthropic) -> None:
        captured = install_fake_anthropic()
        from src.llm.anthropic_client import AnthropicClient

        AnthropicClient(project_id="p").models.generate_content(
            "claude-sonnet-4-6@default", "Q"
        )
        assert captured.captured_kwargs["max_tokens"] >= 1024

    def test_custom_max_tokens_overrides(self, install_fake_anthropic) -> None:
        captured = install_fake_anthropic()
        from src.llm.anthropic_client import AnthropicClient

        AnthropicClient(project_id="p").models.generate_content(
            "claude-sonnet-4-6@default", "Q", config={"max_tokens": 256}
        )
        assert captured.captured_kwargs["max_tokens"] == 256

    def test_system_instruction_propagates(self, install_fake_anthropic) -> None:
        captured = install_fake_anthropic()
        from src.llm.anthropic_client import AnthropicClient

        AnthropicClient(project_id="p").models.generate_content(
            "claude-sonnet-4-6@default",
            "Q",
            config={"system_instruction": "あなたは法律の専門家です"},
        )
        assert captured.captured_kwargs["system"] == "あなたは法律の専門家です"

    def test_no_system_key_when_absent(self, install_fake_anthropic) -> None:
        captured = install_fake_anthropic()
        from src.llm.anthropic_client import AnthropicClient

        AnthropicClient(project_id="p").models.generate_content(
            "claude-sonnet-4-6@default", "Q"
        )
        assert "system" not in captured.captured_kwargs


# ---------------------------------------------------------------------------
# thinking config: デフォルト無効、明示時のみ転送。
# Opus 4.7 / Sonnet 4.6 では temperature は常に抑制される (モデル側の制約)。
# ---------------------------------------------------------------------------
class TestThinkingConfig:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-7@default",
            "claude-opus-4-6@default",
            "claude-sonnet-4-6@default",
            "claude-3-5-sonnet@default",
        ],
    )
    def test_thinking_disabled_by_default(
        self, install_fake_anthropic, model: str
    ) -> None:
        # Layer 1 グラウンディング下では thinking はレイテンシ増のみ大きいため (#10)、
        # 呼び出し側が明示しない限り API に thinking を渡さない。
        captured = install_fake_anthropic()
        from src.llm.anthropic_client import AnthropicClient

        AnthropicClient(project_id="p").models.generate_content(model, "Q")
        assert "thinking" not in captured.captured_kwargs

    def test_thinking_is_forwarded_when_specified(
        self, install_fake_anthropic
    ) -> None:
        captured = install_fake_anthropic()
        from src.llm.anthropic_client import AnthropicClient

        AnthropicClient(project_id="p").models.generate_content(
            "claude-sonnet-4-6@default",
            "Q",
            config={"thinking": {"type": "enabled", "budget_tokens": 1024}},
        )
        assert captured.captured_kwargs.get("thinking") == {
            "type": "enabled",
            "budget_tokens": 1024,
        }


class TestTemperatureHandling:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-opus-4-7@default",
            "claude-opus-4-6@default",
            "claude-sonnet-4-6@default",
        ],
    )
    def test_temperature_is_dropped_for_restricted_models(
        self, install_fake_anthropic, model: str
    ) -> None:
        # Opus 4.7 / Sonnet 4.6 では temperature を渡すと API が 400 を返すので、
        # config に入っていてもクライアント側で必ず捨てられる必要がある
        captured = install_fake_anthropic()
        from src.llm.anthropic_client import AnthropicClient

        AnthropicClient(project_id="p").models.generate_content(
            model, "Q", config={"temperature": 0.5}
        )
        assert "temperature" not in captured.captured_kwargs

    def test_temperature_is_passed_for_legacy_models(
        self, install_fake_anthropic
    ) -> None:
        # 旧モデル (例: claude-3-5-sonnet) では temperature を転送する
        captured = install_fake_anthropic()
        from src.llm.anthropic_client import AnthropicClient

        AnthropicClient(project_id="p").models.generate_content(
            "claude-3-5-sonnet@default", "Q", config={"temperature": 0.3}
        )
        assert captured.captured_kwargs.get("temperature") == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# レスポンスの content block 整形
# ---------------------------------------------------------------------------
class TestResponseAssembly:
    def test_concatenates_multiple_text_blocks(
        self, install_fake_anthropic
    ) -> None:
        install_fake_anthropic([
            _FakeTextBlock("part1 "),
            _FakeTextBlock("part2"),
        ])
        from src.llm.anthropic_client import AnthropicClient

        result = AnthropicClient(project_id="p").models.generate_content(
            "claude-sonnet-4-6@default", "Q"
        )
        assert result.text == "part1 part2"

    def test_skips_non_text_blocks(self, install_fake_anthropic) -> None:
        # adaptive thinking では thinking ブロックが返ることがある。
        # それを最終テキストに含めてしまうと法令判定 JSON が壊れる。
        install_fake_anthropic([
            _FakeNonTextBlock(),
            _FakeTextBlock('{"judgement":"違反"}'),
        ])
        from src.llm.anthropic_client import AnthropicClient

        result = AnthropicClient(project_id="p").models.generate_content(
            "claude-opus-4-7@default", "Q"
        )
        assert result.text == '{"judgement":"違反"}'

    def test_empty_content_returns_empty_string(
        self, install_fake_anthropic
    ) -> None:
        install_fake_anthropic([])
        from src.llm.anthropic_client import AnthropicClient

        result = AnthropicClient(project_id="p").models.generate_content(
            "claude-sonnet-4-6@default", "Q"
        )
        assert result.text == ""
