"use client";

import { useEffect, useMemo, useState } from "react";
import {
  fetchModels,
  fetchModes,
  postJudge,
  type JudgeResponse,
  type ModeInfo,
  type ModelInfo,
} from "../lib/api";

const SAMPLE_QUERIES = [
  "75歳の高齢者が普通自転車で歩道を走行しています。これは違反ですか？反則金はいくらですか？",
  "自転車でスマートフォンを手に持ちながら運転しました。反則金はいくらですか？",
  "自転車で酒気帯び運転をしました。反則金はいくらですか？",
];

export default function JudgeForm() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modes, setModes] = useState<ModeInfo[]>([]);
  const [bootError, setBootError] = useState<string | null>(null);

  const [query, setQuery] = useState(SAMPLE_QUERIES[0]);
  const [modelKey, setModelKey] = useState("flash");
  const [mode, setMode] = useState<ModeInfo["key"]>("layer1");

  const [loading, setLoading] = useState(false);
  const [elapsedSec, setElapsedSec] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<JudgeResponse | null>(null);

  useEffect(() => {
    if (!loading) {
      setElapsedSec(0);
      return;
    }
    const t = setInterval(() => setElapsedSec((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [loading]);

  useEffect(() => {
    Promise.all([fetchModels(), fetchModes()])
      .then(([m, mo]) => {
        setModels(m);
        setModes(mo);
      })
      .catch((e: unknown) => {
        const msg = e instanceof Error ? e.message : String(e);
        setBootError(`API に接続できませんでした: ${msg}`);
      });
  }, []);

  const selectedModel = useMemo(
    () => models.find((m) => m.key === modelKey),
    [models, modelKey],
  );

  const webSearchDisabled =
    !!selectedModel && !selectedModel.supports_web_search;

  // モデル変更で web_search が使えなくなったら layer1 に落とす
  useEffect(() => {
    if (mode === "web_search" && webSearchDisabled) {
      setMode("layer1");
    }
  }, [mode, webSearchDisabled]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await postJudge({ query: query.trim(), model: modelKey, mode });
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6 px-4 py-8">
      <header className="space-y-1">
        <h1 className="text-2xl font-bold tracking-tight">
          Velo-Verify-Gemini
        </h1>
        <p className="text-sm text-slate-600">
          自転車青切符（2026年4月施行）の合法 / 違反をハイブリッド判定
        </p>
      </header>

      {bootError ? (
        <div className="rounded border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
          {bootError}
          <div className="mt-1 text-xs">
            バックエンドを <code>cd backend && python -m src.api.server</code> で起動してください。
          </div>
        </div>
      ) : null}

      <form
        onSubmit={onSubmit}
        className="space-y-4 rounded-lg border border-slate-200 bg-white p-5 shadow-sm"
      >
        <div>
          <label
            htmlFor="query"
            className="mb-1 block text-sm font-medium text-slate-700"
          >
            検索文（シナリオ）
          </label>
          <textarea
            id="query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={3}
            className="w-full rounded border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
            placeholder="例: 自転車で赤信号を無視しました。反則金はいくら？"
          />
          <div className="mt-1 flex flex-wrap gap-2">
            {SAMPLE_QUERIES.map((q) => (
              <button
                type="button"
                key={q}
                onClick={() => setQuery(q)}
                className="rounded-full border border-slate-300 px-2 py-0.5 text-xs text-slate-600 hover:bg-slate-100"
              >
                {q.length > 40 ? q.slice(0, 40) + "…" : q}
              </button>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div>
            <label
              htmlFor="model"
              className="mb-1 block text-sm font-medium text-slate-700"
            >
              モデル
            </label>
            <select
              id="model"
              value={modelKey}
              onChange={(e) => setModelKey(e.target.value)}
              className="w-full rounded border border-slate-300 px-3 py-2 text-sm"
            >
              {models.length === 0 ? (
                <option value="flash">flash</option>
              ) : (
                models.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.label} ({m.model_id})
                  </option>
                ))
              )}
            </select>
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium text-slate-700">
              グラウンディング
            </label>
            <div className="space-y-1">
              {modes.length === 0 ? (
                <p className="text-xs text-slate-500">読み込み中…</p>
              ) : (
                modes.map((m) => {
                  const disabled =
                    m.key === "web_search" && webSearchDisabled;
                  return (
                    <label
                      key={m.key}
                      className={`flex items-start gap-2 text-sm ${
                        disabled ? "text-slate-400" : "text-slate-800"
                      }`}
                    >
                      <input
                        type="radio"
                        name="mode"
                        value={m.key}
                        checked={mode === m.key}
                        disabled={disabled}
                        onChange={() => setMode(m.key)}
                        className="mt-0.5"
                      />
                      <span>
                        {m.label}
                        {disabled ? (
                          <span className="ml-1 text-xs text-slate-400">
                            （Gemini モデルのみ対応）
                          </span>
                        ) : null}
                      </span>
                    </label>
                  );
                })
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <p className="text-xs text-slate-500">
            {loading
              ? `LLM 応答待ち… ${elapsedSec}s（モデルにより 30〜90 秒程度）`
              : "※ LLM 推論には 30 秒以上かかることがあります"}
          </p>
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="rounded bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:bg-slate-300"
          >
            {loading ? `判定中… ${elapsedSec}s` : "判定する"}
          </button>
        </div>
      </form>

      {error ? (
        <div className="rounded border border-rose-300 bg-rose-50 p-3 text-sm text-rose-900">
          {error}
        </div>
      ) : null}

      {result ? <ResultCard result={result} /> : null}
    </div>
  );
}

function ResultCard({ result }: { result: JudgeResponse }) {
  const verdictColor =
    result.judgement === "合法"
      ? "bg-emerald-100 text-emerald-800 border-emerald-300"
      : result.judgement === "違反"
        ? "bg-rose-100 text-rose-800 border-rose-300"
        : "bg-slate-100 text-slate-700 border-slate-300";

  return (
    <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <header className="flex flex-wrap items-center gap-2">
        <span
          className={`rounded border px-2 py-1 text-sm font-bold ${verdictColor}`}
        >
          {result.judgement || "判定不能"}
        </span>
        <span className="text-xs text-slate-500">
          {result.model} · {modeLabel(result.mode)} · {result.response_time_ms}ms
        </span>
      </header>

      <dl className="grid grid-cols-1 gap-3 text-sm md:grid-cols-2">
        <div>
          <dt className="font-medium text-slate-600">根拠条文</dt>
          <dd className="mt-0.5 whitespace-pre-wrap text-slate-900">
            {result.article || "—"}
          </dd>
        </div>
        <div>
          <dt className="font-medium text-slate-600">反則金</dt>
          <dd className="mt-0.5 whitespace-pre-wrap text-slate-900">
            {result.fine || "—"}
          </dd>
        </div>
      </dl>

      <div>
        <h3 className="text-sm font-medium text-slate-600">理由</h3>
        <p className="mt-0.5 whitespace-pre-wrap text-sm text-slate-900">
          {result.reasoning || result.raw_answer}
        </p>
      </div>

      {result.layer1 ? <Layer1Section layer1={result.layer1} /> : null}
      {result.sources.length > 0 ? (
        <SourcesSection sources={result.sources} />
      ) : null}

      <details className="text-xs text-slate-600">
        <summary className="cursor-pointer">LLM 生レスポンス</summary>
        <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-slate-50 p-2 text-slate-700">
          {result.raw_answer}
        </pre>
      </details>
    </section>
  );
}

function Layer1Section({
  layer1,
}: {
  layer1: NonNullable<JudgeResponse["layer1"]>;
}) {
  return (
    <div className="rounded border border-slate-200 bg-slate-50 p-3 text-xs">
      <h3 className="mb-1 text-sm font-semibold text-slate-700">
        Layer 1 グラウンディング（決定論的パース）
      </h3>
      <ul className="mb-2 space-y-0.5">
        {layer1.matches.map((m) => (
          <li key={m.rank} className="font-mono">
            #{m.rank} cos={m.score.toFixed(4)} · {m.article_title}{" "}
            {m.article_caption}
          </li>
        ))}
      </ul>
      {layer1.logic_flags.length > 0 ? (
        <p className="mb-1">
          論理フラグ: <code>{layer1.logic_flags.join(", ")}</code>
        </p>
      ) : null}
      <p className="whitespace-pre-wrap">{layer1.fine_info}</p>
    </div>
  );
}

function SourcesSection({ sources }: { sources: JudgeResponse["sources"] }) {
  return (
    <div className="rounded border border-slate-200 bg-slate-50 p-3 text-xs">
      <h3 className="mb-1 text-sm font-semibold text-slate-700">
        Web Search ソース
      </h3>
      <ul className="space-y-0.5">
        {sources.map((s, i) => (
          <li key={`${s.uri}-${i}`} className="break-all">
            {s.uri ? (
              <a
                href={s.uri}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-700 hover:underline"
              >
                {s.title || s.uri}
              </a>
            ) : (
              <span>{s.title}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function modeLabel(mode: JudgeResponse["mode"]): string {
  switch (mode) {
    case "llm_only":
      return "LLM 単体";
    case "layer1":
      return "Layer 1 グラウンディング";
    case "web_search":
      return "Web Search グラウンディング";
  }
}
