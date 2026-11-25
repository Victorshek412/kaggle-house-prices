# 交功課成套嘢 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal：** 喺 2026-11-29（星期日）23:59 之前，交齊 brief 要求嘅所有嘢：報告文件、8 分鐘片、個人答案、AI 使用說明，打包成用隊名命名嘅 zip。

**Architecture：** 報告初稿已經有（`report/House_Prices_Report_draft.docx`）。Claude 負責起草文字類嘅嘢（AI 說明、個人答案提示、slides、講稿）同做檢查；學生負責讀筆記、改報告、錄片同交；隊友負責 Kaggle 隊名、自己嘅個人答案，同埋片入面自己嗰段。每個 task 都有「點樣知做完」嘅檢查。

**Tech Stack：** Word（.docx → PDF）、Slides artifact（可以下載做 .pptx）、PowerPoint「錄製投影片」或者 Zoom 錄影、HandBrake（壓縮片）、Kaggle 網頁。

**Spec：** 課程 brief（學生喺對話入面貼嘅原文，要求已經逐條抄入下面嘅 Global Constraints）＋ `docs/superpowers/specs/2026-11-22-house-prices-redo-design.md`（§2 寫明報告、slides、片、AI 說明、個人答案會另外處理）。

## Global Constraints（抄自 brief）

- Due: Sunday by 23:59（2026-11-29）。
- "Your submission should be in a zip file and name it according to your team's name."
- "One submission per team (by one representative of the team)."
- Video: "presented by all members with face shown"；"Should be no more than 8 minutes"；"File size no more than 600MByte (please try to reduce the resolution to make it)"；"You may prepare concise slides to do it."
- Document 要有：最有用嘅 feature engineering 同原因；"describe the case your team has obtained the best improvement (not absolute performance)"；"A brief written report (around 10 pages)"；"Screen capture of your team name and final score in Kaggle"；每個成員獨立回答三條問題。
- 個人答案（"by every member of the team"）："What is your main role in doing this competition?"、"What were the main difficulties encountered by the whole team? and by you?"、"What have you learned from this competition?"
- 報告格式：Times New Roman 12（學生指定）。
- AI 使用說明：課程鼓勵用 AI，但要另外交文件交代點用（實際要求要喺課程網頁確認）。

## Review Focus

1. **隊友冇露面**：brief 要求所有成員露面。最遲星期三要約好隊友錄片時間（Task 7 Step 1）。
2. **片超時或者太大**：超過 8:00 或者 600MB 會出事（Task 7 Step 4）。
3. **Kaggle 冇隊名**：而家 S1–S7 係用你個人帳戶交嘅，screenshot 一定要見到隊名（Task 1）。
4. **報告數字同結果唔一致**：你改報告嗰陣可能改錯數字（Task 2 Step 4 會逐個數字對返結果檔案）。
5. **AI 說明同報告第 2 節講法唔一致**（Task 3 Step 3）。

---

## 時間表（香港時間）

| 日子 | 你 | 隊友 | Claude |
|---|---|---|---|
| 日 22（今晚） | Task 1：改隊名、邀請隊友 | 接受 Kaggle 邀請 | — |
| 一 23 | 讀筆記 Part 0–2 | — | Task 3、Task 4 提示、Task 5 slides 初稿 |
| 二 24 | 讀筆記 Part 3–4 | — | Task 6 講稿初稿 |
| 三 25 | 讀筆記 Part 5–6；約好錄片時間 | 確認錄片時間 | 答你嘅問題 |
| 四 26 | Task 2：改完報告、出 PDF | 寫自己嘅個人答案 | Task 2 Step 4 數字檢查 |
| 五 27 | 改 slides 同講稿、練習 | 練習自己嗰段 | 根據你嘅修改更新 slides |
| 六 28 | Task 7：一齊錄片、壓縮 | 一齊錄片 | 檢查片長同大細 |
| 日 29 | Task 8：中午之前打包、交 | 交自己嘅個人答案 | 最後檢查 |

星期日只留做打包同交，唔好留任何未完成嘅嘢到最後一日。

---

### Task 1：Kaggle 隊名同 screenshot（你 + 隊友）

**Files：** Create `report/assets/kaggle_screenshot.png`

- [ ] **Step 1：** Kaggle 比賽頁 → **Team** 分頁 → 改 Team Name（用你哋打算用嚟命名 zip 嘅隊名）。
- [ ] **Step 2：** 喺同一頁邀請隊友合併（Merge）。隊友要先撳 Join Competition、接受比賽規則，然後接受邀請。如果合併唔到，都至少要有隊名，並喺報告封面寫齊兩個成員。
- [ ] **Step 3：** 去 Leaderboard，搵到你哋嗰行（顯示隊名同 0.11624），截圖存做 `report/assets/kaggle_screenshot.png`。
- [ ] **Step 4（檢查）：** 張圖要同時見到**隊名**同 **0.11624**。

### Task 2：讀筆記、改報告、出 PDF（你，Claude 檢查）

**Files：** Modify `report/House_Prices_Report_draft.docx` → 另存 `report/<隊名>_Report.docx` 同 `.pdf`

**Interfaces：** Consumes Task 1 嘅 screenshot、Task 4 嘅個人答案（附錄 B）。

- [ ] **Step 1：** 讀 `v2/notes/part0_basics.md` 至 `part6_error_analysis.md`。有唔明就截圖問 Claude。
- [ ] **Step 2：** 改報告：每段用你自己嘅說話改寫；解釋唔到嘅句子就刪走或者問清楚先。
- [ ] **Step 3：** 填晒所有方括號：封面（隊名、成員、學號、科目）、screenshot、附錄 A 兩條 link、附錄 B 個人答案。
- [ ] **Step 4（檢查，Claude 做）：** 你上載改好嘅 .docx，Claude 抽出入面所有小數，同 `v2/results/cv_results.csv` 同 `v2/results/kaggle_log.csv` 對比，列出所有對唔上嘅數字。
- [ ] **Step 5（檢查）：** 另存 PDF，然後確認：大約 10 頁（附錄除外）；字體係 Times New Roman 12；全文冇剩低任何「[」。

### Task 3：AI 使用說明（Claude 起草，你改）

**Files：** Create `report/AI_use_statement.docx`（1 至 2 頁，Times New Roman 12）

- [ ] **Step 1：** 你喺課程網頁確認呢份文件嘅要求（字數、格式、要答嘅問題），話俾 Claude 知。
- [ ] **Step 2：** Claude 起草，內容只寫有記錄嘅事實：
  - v1 係用 Codex 整嘅（11 月 9 日）；
  - v2 用 Claude 做：解釋概念、寫 code 同逐行解釋、寫學習筆記、起草報告；
  - 你自己做咗咩：讀同改筆記、喺 Kaggle 跑 notebook、決定交邊幾個 submission、改寫報告；
  - 點樣驗證 AI 嘅輸出：自動 test、CV、只交一次 S7。
- [ ] **Step 3（檢查）：** 同報告第 2 節對讀，兩份文件講嘅 AI 角色要一致。

### Task 4：個人答案（你、隊友各自寫）

**Files：** Create `report/Individual_<你個名>.docx`；隊友交佢自己嘅

- [ ] **Step 1：** Claude 俾你每條問題嘅提示，同埋對話記錄入面你真係做過、真係遇過嘅事實清單，例如 outlier bug、CatBoost clone 錯誤、CV 同 Kaggle 唔一致、由零學 pandas。Claude **唔會**代你寫答案。
- [ ] **Step 2：** 你用自己嘅說話寫三條答案。
- [ ] **Step 3：** 隊友自己寫佢嗰份。Claude 唔代寫。
- [ ] **Step 4（檢查）：** 每份都有齊三條答案；「全隊嘅困難」同「自己嘅困難」要分開答。

### Task 5：Slides（Claude 起草，你改）

**Files：** Slides artifact（下載做 `report/<隊名>_Slides.pptx`）

**Interfaces：** Consumes 報告結構（摘要、第 2–9 節）同 `v2/results/figures/*.png`。

- [ ] **Step 1：** Claude 整大約 10 至 12 張 slides，次序跟報告：
  1. 題目同最終分數
  2. 任務同評分
  3. 第一次嘗試、點解重做
  4. EDA
  5. 驗證方法
  6. Feature engineering 樓梯（Q1）
  7. Model 比較
  8. Blending
  9. CV vs Kaggle
  10. Id 2550 同 Step 13
  11. 最大進步（Q2）
  12. 結論
- [ ] **Step 2：** 每張 slide 最多 3 至 4 行字，主要用圖。
- [ ] **Step 3：** 你喺星期五改用字同次序。

### Task 6：講稿同分工（Claude 起草，你 + 隊友練習）

**Files：** Create `report/Video_script.docx`

- [ ] **Step 1：** Claude 寫英文講稿，大約 900 至 950 字（每分鐘大約 125 字，即係 7.5 分鐘），每張 slide 一段，並標明邊個講。
- [ ] **Step 2：** 分工（一人一半）：隊友講 slides 1 至 7（題目、任務、第一次嘗試、EDA、validation、feature engineering，約 484 字）；你講 slides 8 至 14（models、blend、leaderboard、error analysis、Step 13、結果、lessons，約 457 字）。
- [ ] **Step 3（檢查）：** 你同隊友各自讀一次，計時；兩段加埋唔可以超過 7:30，留返 30 秒緩衝。

### Task 7：錄片同壓縮（你 + 隊友）

**Files：** Create `<隊名>_Video.mp4`

- [ ] **Step 1：** 最遲星期三約好錄片時間。方法二揀一：PowerPoint「錄製投影片」（開鏡頭），或者 Zoom 會議分享畫面、兩個人都開鏡頭再錄影。
- [ ] **Step 2：** 錄片：每個人講自己嗰段嗰陣，個樣要見到。
- [ ] **Step 3：** 如果超過 600MB，用 HandBrake 嘅 Fast 720p30 preset 壓縮。
- [ ] **Step 4（檢查）：** 片長 ≤ 8:00；檔案大細 ≤ 600MB；兩個人都有露面同講嘢。有 ffprobe 嘅話，可以用呢句檢查：

```bash
ffprobe -v error -show_entries format=duration,size -of default=nw=1 <隊名>_Video.mp4
```

Expected：`duration` ≤ 480，`size` ≤ 600000000。

### Task 8：打包同交（你）

**Files：** Create `<隊名>.zip`

- [ ] **Step 1：** zip 入面放：`<隊名>_Report.pdf`（入面已經有 Q1、Q2、報告、screenshot、附錄個人答案）、`<隊名>_Video.mp4`、`AI_use_statement.pdf`，同埋（如果需要）`<隊名>_Slides.pdf`。
- [ ] **Step 2（檢查）：** `unzip -l <隊名>.zip` 見到上面所有檔案；zip 檔名就係隊名。
- [ ] **Step 3：** 星期日**中午之前**由你（代表）上載。
- [ ] **Step 4：** 你同隊友各自喺個人提交位交個人答案（如果課程要求另外交）。
