# House Prices 重做：設計文件

日期：2026-11-22 ・ 截止：2026-11-29（星期日）23:59

## 1. 目的同原因

呢份係 Kaggle「House Prices: Advanced Regression Techniques」課程功課，佔科目 25%。評分比重：
- leaderboard 分數 10%
- pathway to solution（feature engineering + prediction）60%
- presentation（片 + slides + 文件）30%

brief 要求交嘅文件（片以外）：
1. 邊項 feature engineering 最有用，同埋點解；
2. 如果 submit 咗多過一次，描述進步最大嗰次（睇進步幅度，唔係絕對分數）；
3. 大約 10 頁報告，講點樣行到個 solution 同主要發現，可以附 Kaggle notebook；
4. Kaggle 隊名同最終分數嘅 screenshot；
5. 每位成員獨立回答：主要角色、全隊同自己遇到嘅困難、學到啲咩。

所以 notebook 嘅結果表同圖要直接答到 1 同 2；學習筆記入面「講師可能會問」嗰部份，要圍繞呢幾條真問題嚟寫。

repo 入面已經有一套做好嘅方案（CV 0.121），但寫得太工程化：自訂 sklearn class、大量 validation check。學生睇唔明，亦講唔出點解咁做。佢有 data mining 底，覺得同同學脫節，好焦慮。

**今次重做嘅目的係令學生 100% 明白成條 pathway**，到時可以喺片、報告同個人問題入面用自己嘅說話解釋每一步。分數係次要。

## 2. 目標、非目標、反目標

**目標**
- 一本自給自足嘅 Kaggle notebook（英文），由 baseline 一步一步行到 CatBoost + blend，每步都有 CV 分數。
- 每個 Part 配一份廣東話學習筆記：講概念、逐行解釋 code、教點睇結果、列出講師可能會問嘅問題。
- 大約 6 次 Kaggle submission，數據足夠答「邊次進步最大」嗰條問題。
- 一張「進步樓梯」圖同一個 CV 結果表，可以直接放入報告同 slides。

**非目標**
- 追 leaderboard 頂尖分數，例如 stacking、大量調參、外部數據。
- 重用舊 code 嘅工程化寫法（自訂 class、provenance、hash check）。
- 今份文件唔包報告、slides、片、AI 使用說明文件同個人答案。呢啲之後另外再傾。

**反目標**（即使行得通都算失敗）
- 學生只係「睇明」，但解釋唔到某一行點解要寫。
- 有任何一步講唔出概念上嘅原因，只係因為「分數好咗」。
- 為咗靚分整出 data leakage，或者講唔清楚 CV 點樣避開 leakage。

## 3. 限制

- 寫 code 只有兩三日（大約 11/23–11/25），之後要寫報告同錄片。
- 運行環境係 **Kaggle Notebook**：數據已經掛好喺 `/kaggle/input/house-prices-advanced-regression-techniques/`，CatBoost 已經預裝。notebook 唔可以 import repo 入面嘅 `.py`。
- Kaggle 每日 submission 有上限，submission 要分開幾日交。
- 讀 CSV 時要保留 `MasVnrType` 嘅 `"None"` 做一個真類別，只當 `NA` 同空白做缺失值：`keep_default_na=False, na_values=["NA", ""]`。
- 課程鼓勵用 AI，但之後要另外交一份文件交代點樣用。所以呢個對話同啲學習筆記本身就係證據，要保留。
- 學生有一個隊友，基本上 free-ride。brief 規定片入面所有成員都要露面。

## 4. 考慮過嘅做法

**重做到幾深**
- A：全新、簡單嘅 notebook，只用 pandas + sklearn。
- B：保留舊 code，逐行學。最快，但難解釋，而且唔係自己做。
- **C（已揀）**：先做 A，最後再加 CatBoost 同 blend。學習同分數兼顧，代價係多半日到一日。

**點樣合作**
- 學生自己填骨架：學得最多，但兩三日唔夠。
- Claude 寫好全部，學生自己睇：快，但容易變成被動學習。
- **已揀**：Claude 寫 code，之後逐行解釋每一行點解要寫，概念要講到學生 100% 明。一次只做一個 Part，學生明晒先去下一個。

## 5. Folder 結構（學生要求兩個版本清楚分開）

```
v1_reference/     ← 舊方案全部搬入嚟（git mv，保留歷史），唔再改
v2/
  house_prices_v2.ipynb
  notes/part1_eda.md … part5_blend.md      （廣東話學習筆記）
  results/        ← cv_results.csv、圖、submission CSV、kaggle_log.csv
README.md         ← 改寫：講清楚 v1 / v2 係咩
```

Claude 的決定：用 `git mv`，舊方案嘅歷史唔會唔見；root 只留 README，咁就唔會兩邊撈亂。

## 6. Pathway（notebook 內容）

**所有步驟共用同一套設定**
- 5-fold `KFold(shuffle=True, random_state=42)`。
- 分數用 log 樓價嘅 RMSE，同 Kaggle 一樣。
- 每一步將結果加入同一個表，欄位包括：step、改咗咩、CV RMSE、同上一步比較。

**預處理嘅分工（核心概念：leakage）**
- `add_features(df)` 用 pandas 寫，只做逐行計算（加總、減法、map），唔用任何整欄統計，所以可以安全咁同時套落 train 同 test。
- 填 median、標準化、one-hot 呢啲要「睇成欄數據」先做到嘅步驟，全部放入 sklearn `Pipeline` + `ColumnTransformer`，喺 CV 入面每個 fold 各自 fit。

呢個分工本身就係要教嘅重點：點解有啲步驟一定要放入 Pipeline。

| Part | Step | 做咩 | Submit? |
|---|---|---|---|
| 1 EDA | – | 樓價分佈同偏斜、缺失值同佢哋嘅意思（NA = 冇車房）、大屋平賣 outlier、同樓價最相關嘅特徵。出 3–4 幅報告用嘅圖 | |
| 2 Baseline | 0 | 估平均價（對照組） | |
| | 1 | Ridge，原始價錢做 target，只做最基本嘅預處理 | ✓ S1 |
| 3 Feature engineering | 2 | 改做預測 log(SalePrice)，預測完再 exp 返去 | |
| | 3 | 缺失值按意思填（NoGarage / NoBasement 等等） | |
| | 4 | 新特徵：TotalSF、TotalBathrooms、HouseAge、RemodelAge | ✓ S2 |
| | 5 | 偏斜嘅面積欄取 log1p（預計最大進步） | ✓ S3 |
| | 6 | 拎走 outlier：GrLivArea > 4000 而且 SalePrice < 300000。只喺 training fold 入面拎走，validation 照計分 | ✓ S4 |
| 4 Model 比較 | 7 | Ridge / Lasso / ElasticNet，用 CV 試一小組 alpha | |
| | 8 | Random Forest（課堂學過，做過渡） | |
| | 9 | CatBoost，直接處理類別欄，唔使 one-hot | ✓ S5 |
| 5 Blend | 10 | 最好嘅線性 model 同 CatBoost 各佔 50% 平均。用固定權重，避免喺 CV 上 overfit 個權重 | ✓ S6（最終） |

**同舊方案唔同嘅地方**（Claude 的決定，學生已經睇過）
- Step 2（log target）獨立成一步，等效果睇得到。
- 加入 Random Forest 做過渡。
- 唔做 XGBoost 同 SVR，控制範圍。

**Step 1 嘅陷阱**：用原始價錢訓練嘅 Ridge 可能估出負數，計 log RMSE 之前要設一個正數下限（clip）。呢個陷阱本身就係講 Step 2 嘅好材料。

最尾：「進步樓梯」圖（每個 step 嘅 CV 分數），加上一個 Kaggle 分數記錄表，交咗 submission 之後由學生填。

## 7. 學習筆記（每個 Part 一份，廣東話）

固定格式：
1. **概念**：直覺為主，只喺需要嘅地方先加一條簡單公式（RMSE、Ridge penalty），唔做推導。
2. **逐行解釋**：每一格 code 每一行點解要寫，冇咗會點。
3. **點睇結果**：個數字代表咩，同上一步比較又代表咩。
4. **講師可能會問**：附答案，順便準備錄片同個人問題。

## 8. 合作節奏同驗證

- 一次交付一個 Part：notebook 新加嘅幾格 + 學習筆記。學生喺 Kaggle 跑，貼返結果，問到明晒先去下一個 Part。
- **Claude 喺本地跑過先交付**，確保 code 行得通、個數字同筆記講嘅一致。
  - 本地要有數據：用 Kaggle API token（經 secure credential 提供），或者學生上載 `train.csv` / `test.csv`。
  - 數據唔會 commit（已經喺 `.gitignore`）。
- 時間表：
  - 11/23：Part 1–2
  - 11/24：Part 3
  - 11/25：Part 4–5
  - 11/26–29：報告、slides、片（另外傾）

## 9. 隊友負責

- 開 Kaggle team、設隊名，最後 screenshot 隊名同分數
- 依學生寫好嘅內容整 slides 排版
- 剪片、壓縮到 600MB 以下、zip 用隊名命名
- 佢自己嘅個人問題
- 片入面至少講一段

## 10. 留俾 builder 決定

- 每格 code 點拆、變數名、圖嘅風格
- alpha 試邊幾個值、Random Forest 同 CatBoost 嘅參數，只要合理同跑得快就得
- notebook 點交付俾學生，例如每個 Part 更新同一本 `.ipynb`，學生 copy 新嘅幾格入 Kaggle
- 預計結果數字唔使同舊方案一樣，以實際跑出嚟嘅為準
