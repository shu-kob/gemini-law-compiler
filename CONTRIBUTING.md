# Contributing Guide

## 開発環境

```bash
pip install -e '.[dev]'
python -m pytest tests/
```

CI (GitHub Actions) は Python 3.11 / 3.12 / 3.13 で `pytest tests/` を実行する。PR 作成時に自動で走るので、マージ前に pass させること。

## 仕様書とテストの同期

`docs/unit_test_spec.md` は各モジュールの振る舞い契約を**言葉で**記述し、`tests/` 配下のテストコードはそれを**実行可能な形で**検証する。両者は同じ内容の二重表現になっているため、実装を変更した際は必ず同時に更新する。

例: `src/parser/legal_compiler.py::_detect_logic_flags` の正規表現を literal match から活用対応に拡張する場合

- テストだけ更新 → spec の記述と食い違い、仕様書が嘘になる
- spec だけ更新 → テストが旧仕様のまま通り、CI の緑が意味を失う
- 実装だけ更新 → 既存テスト (`test_obligation_pattern_is_literal` 等、挙動を固定化しているもの) が赤になる

どのケースでも片方だけ更新するとノイズが生まれる。**実装・テスト・spec の 3 点を同じ PR で更新する**のが原則。

## 対象範囲の切り分け

Unit Test は純ロジックのみを対象とし、Gemini API 呼び出し部分は含めない（End-to-End 実行で検証）。対象範囲の詳細は [`docs/unit_test_spec.md`](docs/unit_test_spec.md) 参照。
