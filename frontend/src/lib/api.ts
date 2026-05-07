export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export type ModelInfo = {
  key: string;
  model_id: string;
  label: string;
  supports_web_search: boolean;
};

export type ModeInfo = {
  key: "llm_only" | "layer1" | "web_search";
  label: string;
};

export type Layer1Match = {
  rank: number;
  score: number;
  article_title: string;
  article_caption: string;
};

export type Layer1Payload = {
  matches: Layer1Match[];
  logic_flags: string[];
  fine_info: string;
  matched_article_text: string;
};

export type GroundingSource = {
  title: string;
  uri: string;
};

export type JudgeResponse = {
  query: string;
  mode: ModeInfo["key"];
  model_key: string;
  model: string;
  judgement: string;
  article: string;
  fine: string;
  reasoning: string;
  raw_answer: string;
  response_time_ms: number;
  sources: GroundingSource[];
  layer1: Layer1Payload | null;
};

export type JudgeRequest = {
  query: string;
  model: string;
  mode: ModeInfo["key"];
};

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const data = await getJson<{ models: ModelInfo[] }>("/api/models");
  return data.models;
}

export async function fetchModes(): Promise<ModeInfo[]> {
  const data = await getJson<{ modes: ModeInfo[] }>("/api/modes");
  return data.modes;
}

export async function postJudge(req: JudgeRequest): Promise<JudgeResponse> {
  const res = await fetch(`${API_BASE}/api/judge`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  const data = await res.json();
  if (!res.ok) {
    const detail = (data && data.detail) || `判定に失敗しました (HTTP ${res.status})`;
    throw new Error(detail);
  }
  return data as JudgeResponse;
}
