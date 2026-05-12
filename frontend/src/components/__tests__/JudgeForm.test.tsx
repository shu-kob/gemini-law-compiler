import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import JudgeForm from "../JudgeForm";
import * as api from "../../lib/api";
import type {
  JudgeResponse,
  ModelInfo,
  ModeInfo,
} from "../../lib/api";

jest.mock("../../lib/api");

const mockedApi = api as jest.Mocked<typeof api>;

const MODELS: ModelInfo[] = [
  { key: "flash", model_id: "gemini-2.5-flash", label: "Flash", supports_web_search: true },
  { key: "sonnet", model_id: "claude-sonnet-4-6", label: "Sonnet", supports_web_search: false },
];

const MODES: ModeInfo[] = [
  { key: "llm_only", label: "LLM 単体" },
  { key: "layer1", label: "Layer 1 グラウンディング" },
  { key: "web_search", label: "Web Search グラウンディング" },
];

const JUDGE_RESPONSE: JudgeResponse = {
  query: "自転車で歩道",
  mode: "layer1",
  model_key: "flash",
  model: "gemini-2.5-flash",
  judgement: "合法",
  article: "第63条の4 普通自転車の歩道通行",
  fine: "—",
  reasoning: "高齢者は例外規定の対象です。",
  raw_answer: "raw answer body",
  response_time_ms: 1234,
  sources: [],
  layer1: {
    matches: [
      { rank: 1, score: 0.8421, article_title: "第63条の4", article_caption: "普通自転車の歩道通行" },
    ],
    logic_flags: ["exception"],
    fine_info: "反則金: 該当なし",
    matched_article_text: "...",
  },
};

beforeEach(() => {
  mockedApi.fetchModels.mockResolvedValue(MODELS);
  mockedApi.fetchModes.mockResolvedValue(MODES);
  mockedApi.postJudge.mockReset();
});

describe("JudgeForm 初期表示", () => {
  it("API から取得した model / mode を選択肢として描画する", async () => {
    render(<JudgeForm />);

    expect(
      screen.getByRole("heading", { name: "Velo-Verify-Gemini" }),
    ).toBeInTheDocument();

    const modelSelect = await screen.findByLabelText("モデル");
    await waitFor(() => {
      expect(
        within(modelSelect as HTMLSelectElement).getByRole("option", {
          name: /Flash/,
        }),
      ).toBeInTheDocument();
    });
    expect(
      within(modelSelect as HTMLSelectElement).getByRole("option", {
        name: /Sonnet/,
      }),
    ).toBeInTheDocument();

    for (const m of MODES) {
      expect(await screen.findByLabelText(new RegExp(m.label))).toBeInTheDocument();
    }
  });

  it("API 呼び出しが失敗したら警告メッセージを表示する", async () => {
    mockedApi.fetchModels.mockRejectedValueOnce(new Error("ECONNREFUSED"));
    render(<JudgeForm />);

    expect(
      await screen.findByText(/API に接続できませんでした: ECONNREFUSED/),
    ).toBeInTheDocument();
  });

  it("デフォルト textarea にサンプル文の 1 つが入っている", async () => {
    render(<JudgeForm />);
    const textarea = screen.getByLabelText("検索文（シナリオ）") as HTMLTextAreaElement;
    expect(textarea.value).toContain("75歳の高齢者");
    await screen.findByRole("option", { name: /Flash/ });
  });
});

describe("JudgeForm モデル選択と Web Search の制約", () => {
  it("web_search 非対応モデルを選ぶと web_search ラジオが disabled になる", async () => {
    const user = userEvent.setup();
    render(<JudgeForm />);

    const modelSelect = (await screen.findByLabelText("モデル")) as HTMLSelectElement;
    // sonnet (supports_web_search=false) に切り替え
    await waitFor(() => expect(modelSelect.options.length).toBeGreaterThanOrEqual(2));
    await user.selectOptions(modelSelect, "sonnet");

    const webSearchRadio = await screen.findByRole("radio", {
      name: /Web Search/,
    });
    expect(webSearchRadio).toBeDisabled();
  });

  it("web_search を選んでいる状態で非対応モデルに切替えると layer1 に戻る", async () => {
    const user = userEvent.setup();
    render(<JudgeForm />);

    const webSearchRadio = await screen.findByRole("radio", {
      name: /Web Search/,
    });
    await user.click(webSearchRadio);
    expect(webSearchRadio).toBeChecked();

    const modelSelect = (await screen.findByLabelText("モデル")) as HTMLSelectElement;
    await user.selectOptions(modelSelect, "sonnet");

    const layer1Radio = await screen.findByRole("radio", { name: /Layer 1/ });
    await waitFor(() => expect(layer1Radio).toBeChecked());
  });
});

describe("JudgeForm サンプルクエリのチップクリック", () => {
  it("チップをクリックすると textarea にそのクエリが入る", async () => {
    const user = userEvent.setup();
    render(<JudgeForm />);
    await screen.findByRole("option", { name: /Flash/ });

    const textarea = screen.getByLabelText("検索文（シナリオ）") as HTMLTextAreaElement;
    const sakeChip = screen.getByRole("button", { name: /酒気帯び/ });
    await user.click(sakeChip);

    expect(textarea.value).toContain("酒気帯び");
  });
});

describe("JudgeForm 送信フロー", () => {
  it("postJudge を選択値で呼び、結果カードに判定結果を描画する", async () => {
    const user = userEvent.setup();
    mockedApi.postJudge.mockResolvedValueOnce(JUDGE_RESPONSE);
    render(<JudgeForm />);

    // models / modes ロードを待つ
    await screen.findByRole("option", { name: /Flash/ });

    const submit = screen.getByRole("button", { name: "判定する" });
    await user.click(submit);

    await waitFor(() => expect(mockedApi.postJudge).toHaveBeenCalledTimes(1));
    expect(mockedApi.postJudge).toHaveBeenCalledWith({
      query: expect.stringContaining("75歳の高齢者"),
      model: "flash",
      mode: "layer1",
    });

    expect(await screen.findByText("合法")).toBeInTheDocument();
    // 「第63条の4 普通自転車の歩道通行」は (a) 根拠条文 dd と (b) Layer1 マッチリストの両方に登場する
    expect(screen.getAllByText(/第63条の4 普通自転車の歩道通行/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/高齢者は例外規定の対象です/)).toBeInTheDocument();
    expect(
      screen.getByText(/Layer 1 グラウンディング（決定論的パース）/),
    ).toBeInTheDocument();
  });

  it("postJudge が reject したら error バナーを表示する", async () => {
    const user = userEvent.setup();
    mockedApi.postJudge.mockRejectedValueOnce(new Error("model not found"));
    render(<JudgeForm />);

    await screen.findByRole("option", { name: /Flash/ });
    await user.click(screen.getByRole("button", { name: "判定する" }));

    expect(await screen.findByText("model not found")).toBeInTheDocument();
  });

  it("textarea が空のときは submit ボタンが disabled", async () => {
    const user = userEvent.setup();
    render(<JudgeForm />);
    // 初期 useEffect の API ロードを待ってから clear する（act 警告回避）
    await screen.findByRole("option", { name: /Flash/ });
    const textarea = screen.getByLabelText("検索文（シナリオ）") as HTMLTextAreaElement;
    await user.clear(textarea);
    expect(screen.getByRole("button", { name: "判定する" })).toBeDisabled();
  });
});
