import {
  fetchModels,
  fetchModes,
  postJudge,
  API_BASE,
  type JudgeResponse,
  type ModelInfo,
  type ModeInfo,
} from "../api";

type FetchInput = Parameters<typeof fetch>[0];
type FetchInit = Parameters<typeof fetch>[1];

const originalFetch = global.fetch;

function mockFetchOnce(body: unknown, init: { ok?: boolean; status?: number } = {}) {
  const { ok = true, status = ok ? 200 : 500 } = init;
  const fn = jest.fn().mockResolvedValue({
    ok,
    status,
    json: async () => body,
  } as Response);
  global.fetch = fn as unknown as typeof fetch;
  return fn;
}

afterEach(() => {
  global.fetch = originalFetch;
});

describe("fetchModels", () => {
  it("GET /api/models から models 配列を取り出して返す", async () => {
    const models: ModelInfo[] = [
      { key: "flash", model_id: "gemini-2.5-flash", label: "Flash", supports_web_search: true },
      { key: "sonnet", model_id: "claude-sonnet-4-6", label: "Sonnet", supports_web_search: false },
    ];
    const fetchMock = mockFetchOnce({ models });

    const result = await fetchModels();

    expect(result).toEqual(models);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [FetchInput, FetchInit];
    expect(url).toBe(`${API_BASE}/api/models`);
    expect(init).toEqual({ cache: "no-store" });
  });

  it("res.ok=false なら Error を投げる", async () => {
    mockFetchOnce({}, { ok: false, status: 503 });
    await expect(fetchModels()).rejects.toThrow("GET /api/models failed: 503");
  });
});

describe("fetchModes", () => {
  it("GET /api/modes から modes 配列を取り出して返す", async () => {
    const modes: ModeInfo[] = [
      { key: "llm_only", label: "LLM 単体" },
      { key: "layer1", label: "Layer 1" },
      { key: "web_search", label: "Web Search" },
    ];
    const fetchMock = mockFetchOnce({ modes });

    const result = await fetchModes();

    expect(result).toEqual(modes);
    const [url] = fetchMock.mock.calls[0] as [FetchInput, FetchInit];
    expect(url).toBe(`${API_BASE}/api/modes`);
  });

  it("res.ok=false なら Error を投げる", async () => {
    mockFetchOnce({}, { ok: false, status: 500 });
    await expect(fetchModes()).rejects.toThrow("GET /api/modes failed: 500");
  });
});

describe("postJudge", () => {
  const judgeRes: JudgeResponse = {
    query: "自転車で歩道",
    mode: "layer1",
    model_key: "flash",
    model: "gemini-2.5-flash",
    judgement: "合法",
    article: "第63条の4",
    fine: "—",
    reasoning: "高齢者は例外",
    raw_answer: "raw",
    response_time_ms: 1234,
    sources: [],
    layer1: null,
  };

  it("JSON ボディで POST /api/judge を呼び、JudgeResponse を返す", async () => {
    const fetchMock = mockFetchOnce(judgeRes);

    const result = await postJudge({
      query: "自転車で歩道",
      model: "flash",
      mode: "layer1",
    });

    expect(result).toEqual(judgeRes);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0] as [FetchInput, FetchInit];
    expect(url).toBe(`${API_BASE}/api/judge`);
    expect(init?.method).toBe("POST");
    expect((init?.headers as Record<string, string>)["Content-Type"]).toBe(
      "application/json",
    );
    expect(JSON.parse(init?.body as string)).toEqual({
      query: "自転車で歩道",
      model: "flash",
      mode: "layer1",
    });
  });

  it("res.ok=false の時、body.detail があればそれをエラーメッセージに使う", async () => {
    mockFetchOnce({ detail: "model not found" }, { ok: false, status: 404 });
    await expect(
      postJudge({ query: "x", model: "ghost", mode: "layer1" }),
    ).rejects.toThrow("model not found");
  });

  it("res.ok=false で detail が無ければ HTTP ステータスを含む汎用メッセージを投げる", async () => {
    mockFetchOnce({}, { ok: false, status: 500 });
    await expect(
      postJudge({ query: "x", model: "flash", mode: "layer1" }),
    ).rejects.toThrow("判定に失敗しました (HTTP 500)");
  });
});
