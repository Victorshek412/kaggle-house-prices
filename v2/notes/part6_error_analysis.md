# Part 6 學習筆記：Error analysis（leaderboard 教識我哋嘅嘢）

> **前置知識（唔識就先睇 Part 0）：** boolean mask 同 `~` → §2.3；`.to_numpy()`、index → §3.1；for 迴圈（兩層）同 `enumerate` → §1.4；dict → §1.2；function 同 docstring → §1.3；`clone`、`fit`、`predict` → §4.2 同 Part 2 筆記。
> 另外會用到 Part 1 嘅 `outlier`、`train`、`test`；Part 2 嘅 `cv_evaluate`、`record`、`save_submission`、`kf`；Part 3 嘅 `X_final`；Part 4 嘅 `X_cat`、`catboost`、`lasso_alpha`、`best_C`、`best_gamma`、`linear_pipeline`、`y_log`、`test_preds`；Part 5 嘅 `blend_oof`、`blend_test`、`summary`、`cv_by_step`。

---

## 0. 點解會有 Part 6

交咗 S1–S6 上 Kaggle 之後，我哋發現兩樣嘢同 CV 唔一致：

1. **S5（CatBoost）喺 CV 好過 S4（Ridge），但喺 Kaggle 差過佢。**
2. **最終 blend（S6，0.12037）差過我哋隊第一次嘅嘗試（0.11745）。**

Part 6 嘅工作就係**查出原因，再作出一個有理據嘅修正（Step 13）**。呢個過程叫 **error analysis（誤差分析）**：唔係盲目試新 model，而係睇清楚「錯喺邊度、點解錯」，再對症下藥。

**一句總結：**
- test 入面有一間好特別嘅屋（Id 2550），佢同 train 入面兩間 outlier 係同一類。
- 我哋喺 Step 6 將嗰兩間 outlier 拎走咗，結果 model 學唔到呢類屋，對 Id 2550 估得好貴。
- 更麻煩嘅係，我哋嘅 CV 剛好有盲點，睇唔到呢個問題。

---

## 1. Cell 1：Kaggle 分數 vs CV

```python
kaggle = pd.DataFrame({
    "submission": ["s1_ridge_raw", "s2_new_features", "s3_log_features", "s4_no_outliers",
                   "s5_catboost", "s6_blend", "first attempt (ElasticNet + CatBoost)"],
    "cv_rmse": [cv_by_step[s] for s in ["1", "4", "5", "6b", "10", "12"]] + [0.12117],
    "kaggle_score": [0.14188, 0.13044, 0.12253, 0.12236, 0.12310, 0.12037, 0.11745],
})
```
- 用一個 dict 砌一張表（→ Part 0 §3），有 3 欄：submission 名、CV 分數、Kaggle 分數。
- `[cv_by_step[s] for s in [...]]`：list comprehension（→ Part 0 §1.8），逐個 step 去 Part 5 整好嘅 `cv_by_step` 度查 CV 分數。
- `+ [0.12117]`：兩個 list 用 `+` 會駁埋，即係喺尾加多一個數。0.12117 係第一次嘗試（v1）嘅 CV 分數，記錄喺 `v1_reference/results/session_summary.json`。
- **Kaggle 分數係人手抄入去嘅**，因為 test 嘅真實樓價藏咗喺 Kaggle，notebook 自己計唔到。

```python
kaggle["kaggle_change"] = kaggle.kaggle_score.diff()
kaggle.loc[6, "kaggle_change"] = np.nan
kaggle.round(5)
```
- `.diff()` 計每行同上一行嘅差（→ Part 0 §3.3），即係每次 submission 喺 Kaggle 進步咗幾多。
- 第 6 行（第一次嘗試）唔屬於 S1–S6 呢條路線，同 S6 相減冇意思，所以設做 NaN。

跑出嚟會見到：

```text
                              submission  cv_rmse  kaggle_score  kaggle_change
0                           s1_ridge_raw  0.15534       0.14188            NaN
1                        s2_new_features  0.15121       0.13044       -0.01144
2                        s3_log_features  0.13239       0.12253       -0.00791
3                         s4_no_outliers  0.12766       0.12236       -0.00017
4                            s5_catboost  0.12320       0.12310        0.00074
5                               s6_blend  0.11906       0.12037       -0.00273
6  first attempt (ElasticNet + CatBoost)  0.12117       0.11745            NaN
```

**點睇：**
- S1 → S4 喺 Kaggle 都係一路跌，同 CV 方向一致，代表 Part 2–3 嘅方法有效。
- **S5 嘅 kaggle_change 係 +0.00074**，即係升咗。但 CV 話 S5 好過 S4（0.1232 vs 0.1277）。
- **S6 喺 CV 係最好（0.1191），但 Kaggle 0.12037 差過第一次嘗試嘅 0.11745。**

**第一個解釋：運氣（noise）。** 我另外計過，同一個 Ridge model，5 個 fold 各自嘅分數由 **0.105 到 0.172** 都有。即係單係「抽到邊 292 間屋」，分數就可以差 0.05 咁多。Kaggle leaderboard 都係另一批屋，所以 0.001 至 0.003 級數嘅差異好大部份係運氣。

不過運氣唔係全部，下面三格搵到一個真正嘅原因。

---

## 2. Cell 2：CV 嘅誤差嚟自邊度

```python
blend_error = y_log - blend_oof
```
Step 12 blend 對每間屋嘅誤差（真實 log 價錢 − 估計），共 1460 個數。

```python
fold_of = np.zeros(len(train), dtype=int)
for fold, (_, valid_idx) in enumerate(kf.split(X), start=1):
    fold_of[valid_idx] = fold
```
- 目的：搵出每間屋係喺第幾個 fold 做驗證。
- `np.zeros(len(train), dtype=int)`：1460 個 0，`dtype=int` 代表整數。
- `kf.split(X)` 同 Part 2 一樣，俾出 5 組（訓練行, 驗證行）。我哋只要驗證行，所以訓練行用 `_` 丟棄。
- `enumerate(..., start=1)`：一路行一路編號，由 1 開始數（→ Part 0 §1.4）。
- `fold_of[valid_idx] = fold`：將呢個 fold 嘅驗證屋全部標上 fold 編號。

```python
is_outlier = outlier.to_numpy()
print(pd.DataFrame({"Id": train.Id[is_outlier].to_numpy(), "fold": fold_of[is_outlier],
                    "true price": train.SalePrice[is_outlier].to_numpy(),
                    "Step 12 log error": blend_error[is_outlier].round(2)}).to_string(index=False))
```
- `outlier` 係 Part 1 嘅 True/False Series；`.to_numpy()` 變成 numpy array，方便同其他 array 一齊用。
- 砌一張細表：兩間 outlier 嘅 Id、喺邊個 fold、真實價錢、Step 12 嘅誤差。
- `.to_string(index=False)`：印嘅時候唔印行號。

```python
print(f"\nStep 12 blend CV RMSE, all 1460 houses:        {np.sqrt(np.mean(blend_error ** 2)):.4f}")
print(f"Step 12 blend CV RMSE, without these 2 houses:  {np.sqrt(np.mean(blend_error[~is_outlier] ** 2)):.4f}")
```
- 第一行：用全部屋計 RMSE。
- 第二行：`blend_error[~is_outlier]` 揀出「唔係 outlier」嘅 1458 間屋，再計一次 RMSE。
- `\n` 喺字串入面代表「轉行」，所以第一行前面會空一行。

跑出嚟會見到：

```text
  Id  fold  true price  Step 12 log error
 524     3      184750              -1.25
1299     3      160000              -1.40

Step 12 blend CV RMSE, all 1460 houses:        0.1191
Step 12 blend CV RMSE, without these 2 houses:  0.1085
```

**點睇：兩個發現**

1. **兩間屋估錯咗好多。** log 誤差 −1.25 即係估貴咗 e^1.25 ≈ 3.5 倍：真實 18.5 萬，估到 60 幾萬。
2. **兩間屋就搞到 RMSE 由 0.1085 升到 0.1191。** 點解兩間屋可以影響咁大？因為 RMSE 係用**平方**：
   - 一間普通屋嘅誤差大約 0.1，平方係 0.01；
   - 呢兩間嘅誤差係 1.25 同 1.40，平方係 1.56 同 1.96，**每間等於 150 至 200 間普通屋**。
   - 計返數：全部 1460 間嘅誤差平方總和大約係 20.7，佢哋兩間已經佔咗 3.5，即係 **0.14% 嘅屋貢獻咗大約 17% 嘅誤差**。
3. **兩間都喺 fold 3。** 即係佢哋係**一齊**做驗證嘅。輪到估佢哋嗰陣，訓練數據入面兩間都冇（另外 Step 6 開始亦都會喺訓練拎走佢哋）。所以 CV **永遠測試唔到**「由其中一間學識、估另一間」呢種做法。呢個就係 CV 嘅**盲點**。

---

## 3. Cell 3：test 入面有第三間

```python
huge_test = (test.GrLivArea > 4000).to_numpy()
print(test.loc[huge_test, ["Id", "GrLivArea", "TotalBsmtSF", "OverallQual", "SaleCondition"]].to_string(index=False))
```
- 搵出 test 入面居住面積大過 4000 呎嘅屋。
- **留意：** 我哋只係睇 test 嘅**特徵**（面積、質素等等），冇睇亦都睇唔到佢嘅價錢。睇 test 特徵係完全合規嘅，Part 1 都睇過 test 嘅缺失值。

```python
for label, predictions in [("S4 Ridge", test_preds["ridge"]), ("S5 CatBoost", test_preds["catboost"]),
                           ("S6 blend", blend_test)]:
    print(f"{label:12s} predicts ${np.exp(predictions[huge_test][0]):,.0f} for Id 2550")
```
- 逐個 submission 攞返佢對呢間屋嘅預測。
- `predictions[huge_test]` 用 mask 揀出嗰一間，`[0]` 攞第一個（亦都係唯一一個）。
- `np.exp` 將 log 變返做價錢。
- `{…:,.0f}`：唔要小數，每三位加一個逗號（例如 763,102）。`{label:12s}` 令文字佔 12 格，對齊。

跑出嚟會見到：

```text
  Id  GrLivArea  TotalBsmtSF  OverallQual SaleCondition
2550       5095       5095.0           10       Partial

S4 Ridge     predicts $763,102 for Id 2550
S5 CatBoost  predicts $564,883 for Id 2550
S6 blend     predicts $642,880 for Id 2550
```

**點睇：**
- Id 2550：5095 呎、質素 10 分、Partial sale。同 train 嘅 524（4676 呎、10 分、Partial）同 1299（5642 呎、10 分、Partial）**幾乎一模一樣**。
- 佢嘅兩個「孖生兄弟」實際賣 16 至 18.5 萬，但我哋嘅 model 估 56 至 76 萬。
- **原因：** Step 6 喺 training 拎走咗佢兩個兄弟，model 從來冇見過「超大 Partial sale 會賣得平」，只學到「大屋 = 貴」。
- **第一次嘗試估 49 萬**，比 S6 平，因為佢嘅 CatBoost 冇拎走 outlier。呢個好可能係佢 Kaggle 分數好啲嘅主要原因之一。不過我哋唔知 Id 2550 真實賣幾錢，所以唔可以 100% 肯定。

---

## 4. Cell 4：Step 13 嘅兩個工具

### 4.1 概念：唔好刪走，而係話俾 model 知

Step 6 嘅做法係「刪走 outlier」。問題係刪走咗，model 就完全唔知有呢類屋。

Step 13 改做：**保留佢哋，加一個特徵 `HugePartial`**：
- 面積 > 4000 呎**而且**係 Partial sale → 1；
- 其他屋 → 0。

**點解加一欄 0/1 就有用？** 以線性 model 為例（→ Part 0 §4.3），佢會幫呢欄學一個 coefficient。因為只有呢類屋係 1，呢個 coefficient 就變成專門畀呢類屋嘅「調整」，例如「減 1.2（log 尺度）」。普通屋呢欄係 0，乘出嚟都係 0，完全唔受影響。

### 4.2 逐行解釋

```python
def clean_years(df):
    """Impossible garage years (e.g. 2207, a typo) become missing."""
    df = df.copy()
    df.loc[df["GarageYrBlt"] > df["YrSold"] + 1, "GarageYrBlt"] = np.nan
    return df
```
- test 有一間屋（Id 2593）嘅車房年份係 **2207**，明顯係 2007 打錯。
- 規則：車房年份遲過「賣出年份 + 1」就係唔可能嘅，改做 NaN（缺失）。之後 Pipeline 會當普通缺失咁填 median，CatBoost 亦識處理 NaN。
- **點解係 +1，唔係直接「大過賣出年份」？** 樓花可以賣咗先起好，所以遲 1 年係合理嘅。例如 Id 2550 嘅車房係 2008 年，佢 2007 年賣。
- **點解唔直接改做 2007？** 因為我哋唔肯定原本係幾多。改做「唔知」最老實，亦唔會亂估。
- `df.copy()` 同 Part 3 一樣，唔改原本嗰張表（→ Part 0 §3.4）。

```python
def add_huge_partial_flag(df, raw):
    """1 for houses over 4,000 sq ft sold as Partial sales, else 0 (uses the raw, unlogged area)."""
    df = df.copy()
    df["HugePartial"] = ((raw["GrLivArea"] > 4000) & (raw["SaleCondition"] == "Partial")).astype(int).to_numpy()
    return df
```
- `raw`：**原始數據**（`train` 或者 `test`）。
- **點解要另外傳入 `raw`，唔直接用 `df` 嘅 GrLivArea？** 因為 `X_final` 嘅 GrLivArea 已經取咗 log1p（最大得 8.6），「> 4000」永遠唔會成立。**呢個正正係 Part 3 §6.3 嗰個 bug 嘅教訓**，今次一開始就避開咗。
- `(條件1) & (條件2)`：兩個都要成立（→ Part 0 §2.3）。
- `.astype(int)`：True → 1，False → 0。
- `.to_numpy()`：變成純數字 array 先放入去，避免 pandas 按 index 對位時出錯。`df` 同 `raw` 嘅行次序係一樣嘅，所以直接放就啱。

```python
X13, X13_test = add_huge_partial_flag(clean_years(X_final), train), add_huge_partial_flag(clean_years(X_test_final), test)
X13_cat, X13_cat_test = add_huge_partial_flag(clean_years(X_cat), train), add_huge_partial_flag(clean_years(X_cat_test), test)
```
- 由內至外讀：先 `clean_years`，再 `add_huge_partial_flag`。
- 兩套特徵都要做：`X_final`（線性 model、SVR、XGBoost 用）同 `X_cat`（CatBoost 用）。
- train 同 test 都用同一個 function，保證做法一樣。

```python
print("flagged in train:", train.Id[X13.HugePartial == 1].tolist(), "  flagged in test:", test.Id[X13_test.HugePartial == 1].tolist())
print("garage years cleaned in test:", test.Id[X_test_final.GarageYrBlt > X_test_final.YrSold + 1].tolist())
```
印出嚟檢查：邊幾間屋被標記、邊幾間屋嘅年份被清理。

跑出嚟會見到：

```text
flagged in train: [524, 1299]   flagged in test: [2550]
garage years cleaned in test: [2593]
```

同預期完全一樣：train 嘅兩間孖生兄弟、test 嘅 2550，同埋年份打錯嘅 2593。

---

## 5. Cell 5：喺 CV 測唔到嘅地方，另外設計驗證

### 5.1 概念：Leave-one-outlier-out

CV 測唔到（兩間 outlier 喺同一個 fold），所以我哋**自己設計一個專門嘅驗證**：
- 留起 524，用其餘 1459 間屋訓練（**包括 1299**），然後估 524。
- 再反過嚟：留起 1299，用其餘（包括 524）訓練，估 1299。
- 每次都比較兩種做法：
  - **現時做法（Step 6–12）：** 兩間 outlier 都唔用，冇標記；
  - **Step 13 做法：** 保留另一間，加標記。

呢個好似考試：考驗 model 識唔識「由一個例子學識，用喺另一個類似嘅例子」。呢個正正係我哋對 test 嘅 Id 2550 嘅期望。

### 5.2 逐行解釋

```python
def make_xgb(seed=42):
    """The XGBoost model from Step 9 (same settings), with a choice of random seed."""
    return linear_pipeline(XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=3,
                                        subsample=0.8, colsample_bytree=0.5, min_child_weight=2,
                                        n_jobs=-1, random_state=seed))
```
- 整一個同 Step 9 **一模一樣設定**嘅 XGBoost，只係 random seed 可以揀，預設係 42，同 Step 9 一樣。
- 寫成 function，因為之後 seed averaging 會用唔同 seed 整幾個。

```python
step13_models = {  # the Step 12 blend members, unchanged
    "lasso": (linear_pipeline(Lasso(alpha=lasso_alpha, max_iter=50000)), "linear"),
    "xgb": (make_xgb(), "linear"),
    "catboost": (catboost, "tree"),
    "svr": (linear_pipeline(SVR(C=best_C, epsilon=0.01, gamma=best_gamma)), "linear"),
}
```
- 一個 dict：model 名 → （model, 用邊套特徵）。
- 4 個 model **同 Step 12 blend 完全一樣**：Lasso 用 Part 4 揀好嘅 `lasso_alpha`，SVR 用 `best_C`、`best_gamma`，CatBoost 直接用 Part 4 個 `catboost` 物件。**冇任何重新調參**。
- `"linear"` 代表用 `X_final` 嗰套特徵（one-hot + log），`"tree"` 代表用 `X_cat` 嗰套（CatBoost 自己處理類別）。

```python
rows = []
for held_out in [524, 1299]:
    is_target = (train.Id == held_out).to_numpy()
```
- 外層迴圈：逐間 outlier 做「被留起嗰間」。
- `is_target`：一條 True/False array，只有被留起嗰間係 True。

```python
    for setting, X_linear, X_tree, keep in [
            ("current: drop both, no flag", X_final, X_cat, ~is_outlier),
            ("Step 13: keep the other + flag", X13, X13_cat, ~is_target)]:
```
- 第二層迴圈：兩種做法，每種一次過拆出 4 樣嘢（unpacking → Part 0 §1.4）：
  - `setting`：名；
  - `X_linear`、`X_tree`：兩套特徵；
  - `keep`：邊啲屋用嚟訓練。
- **關鍵分別喺 `keep`：**
  - 現時做法：`~is_outlier`，即係兩間 outlier 都唔要；
  - Step 13：`~is_target`，即係只係唔要被留起嗰間，另一間保留。

```python
        predictions = []
        for name, (model, kind) in step13_models.items():
            X_use = X_linear if kind == "linear" else X_tree
            fitted = clone(model).fit(X_use[keep], y_log[keep])
            predictions.append(fitted.predict(X_use[is_target])[0])
```
- 第三層迴圈：逐個 model。
- `.items()` 會俾出（key, value）一對；value 本身又係一個 tuple，所以寫 `name, (model, kind)` 一次過拆開。
- `X_use = X_linear if kind == "linear" else X_tree`：一行 if（→ Part 0 §1.8），揀啱嘅特徵。
- `clone(model).fit(X_use[keep], y_log[keep])`：整個全新 model，用 `keep` 揀出嘅屋訓練。`X_use[keep]` 用 True/False array 揀行。
- `.predict(X_use[is_target])[0]`：估被留起嗰間，`[0]` 攞出個數字。
- `.append(...)`：將預測加落 `predictions` list 嘅尾。

```python
        blend = np.mean(predictions)
        rows.append({"held-out Id": held_out, "setting": setting,
                     "true price": int(train.SalePrice[is_target].iloc[0]),
                     "blend predicts": int(round(np.exp(blend))),
                     "log error": round(y_log[is_target][0] - blend, 2)})
pd.DataFrame(rows)
```
- 4 個 model 嘅預測平均，即係 blend，同 Step 12 一樣嘅等權重。
- 每次結果砌成一個 dict，加入 `rows`；最後用 `pd.DataFrame(rows)` 將一串 dict 變成一張表（一個 dict = 一行）。
- `int(...)`：變做整數，咁印出嚟就冇小數。

跑出嚟會見到：

```text
   held-out Id                         setting  true price  blend predicts  log error
0          524     current: drop both, no flag      184750          665634      -1.28
1          524  Step 13: keep the other + flag      184750          242533      -0.27
2         1299     current: drop both, no flag      160000          708551      -1.49
3         1299  Step 13: keep the other + flag      160000          303257      -0.64
```

（notebook 入面張表太闊，log error 會被擠到下一段顯示，內容一樣。）

**點睇：**
- **現時做法：** 524 估 66.6 萬（真實 18.5 萬），1299 估 70.9 萬（真實 16 萬）。log 誤差 −1.28、−1.49，即係估貴咗 3.6 至 4.4 倍。
- **Step 13 做法：** 只要見過另一間，524 就估 24.3 萬，1299 估 30.3 萬。log 誤差收窄到 −0.27、−0.64。
- 誤差嘅平方由 1.64 + 2.22 跌到 0.07 + 0.41，**大約少咗 88%**。
- 仍然唔完美：1299 仲係估貴咗大約 1.9 倍，因為 model 只有一個例子可以學。但已經好咗好多。

**呢個就係 Step 13 嘅證據。** 就算 CV 分數唔變，呢個驗證都證明咗新做法識得處理呢類屋。

---

## 6. Cell 6：Step 13 — 用新特徵重新做 blend

```python
oof13, test13 = {}, {}
for name, (model, kind) in step13_models.items():
    X_train_use, X_test_use = (X13, X13_test) if kind == "linear" else (X13_cat, X13_cat_test)
    score, oof13[name], test13[name] = cv_evaluate(model, X_train_use, y, X_test_use,
                                                   target="log", drop_outliers=False)
    print(f"{name:9s} CV RMSE {score:.5f}")
```
- 同 Part 4 一樣用 `cv_evaluate` 做 5-fold CV，結果存入 `oof13`、`test13` 兩個 dict。
- **唯一分別：`drop_outliers=False`**，即係保留 outlier 落 training。
- 一行 if 揀特徵，不過今次一次過揀兩樣（train 同 test），所以兩邊都係 tuple。

```python
blend13_oof = np.mean(list(oof13.values()), axis=0)
blend13_test = np.mean(list(test13.values()), axis=0)
record("13", "Keep outliers + HugePartial flag, clean garage years; same 4-model blend",
       np.sqrt(np.mean((y_log - blend13_oof) ** 2)))
save_submission("s7_outlier_flag", blend13_test)
```
- `oof13.values()` 攞出 4 個 model 嘅預測；`list(...)` 變成 list；`np.mean(..., axis=0)` 每間屋平均 4 個預測（同 Part 5 一樣）。
- 記錄做 Step 13，出 submission **S7**。

```python
for label, predictions in [("S6 blend (Step 12)", blend_test), ("S7 blend (Step 13)", blend13_test)]:
    print(f"{label}: Id 2550 predicted at ${np.exp(predictions[huge_test][0]):,.0f}")
change = np.abs(blend13_test - blend_test)[~huge_test]
print(f"other test houses: median change {np.median(change):.4f}, largest change {change.max():.4f} (log scale)")
```
- 比較 S6 同 S7 對 Id 2550 嘅預測。
- 再睇**其他 1458 間** test 屋嘅預測變咗幾多：`np.abs` 係絕對值，`[~huge_test]` 揀返 Id 2550 以外嘅屋，`np.median` 係中位數。

跑出嚟會見到：

```text
lasso     CV RMSE 0.12559
xgb       CV RMSE 0.12347
catboost  CV RMSE 0.12318
svr       CV RMSE 0.12328
Step 13: CV RMSE 0.11910 (+0.00004 vs previous step)
saved s7_outlier_flag.csv
S6 blend (Step 12): Id 2550 predicted at $642,880
S7 blend (Step 13): Id 2550 predicted at $259,695
other test houses: median change 0.0025, largest change 0.0621 (log scale)
```

**點睇：**
- **CV 幾乎冇變（0.11906 → 0.11910）。** 呢個係預期之內，原因見 §2：兩間 outlier 喺同一個 fold，CV 根本獎勵唔到呢個做法。報告要講清楚呢點，唔好當成「冇進步」。
- **Id 2550 由 64.3 萬變成 26.0 萬**，同佢兩個孖生兄弟嘅價錢（16 至 18.5 萬）接近好多。
- **其他屋幾乎冇郁：** 中位數只變咗 0.0025（大約 0.25%），最多嘅都只係 0.06。即係 Step 13 只係針對呢類特別嘅屋，冇影響正常嘅屋。

---

## 7. Cell 7：試過但唔採用嘅方法

### 7.1 概念

報告寫「我哋試過 A、B、C，佢哋冇幫助，所以冇用」**係好重要嘅 pathway 內容**。佢證明你唔係隨便揀一個方法，而係比較過先決定。

全部都用同一個 CV、喺 Step 13 嘅特徵上面比較，所以公平。

### 7.2 逐行解釋

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

def rmse(log_predictions):
    return np.sqrt(np.mean((y_log - log_predictions) ** 2))
```
一個細 function：俾一串 OOF 預測，計 RMSE，費事成日寫成條公式。

```python
QUALITY = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}

def add_quality_scores(df, raw):
    df = df.copy()
    for column in ["ExterQual", "ExterCond", "BsmtQual", ...]:
        df[column + "Score"] = raw[column].map(QUALITY).fillna(0).to_numpy()
    df["QualityLivingArea"] = np.log1p(raw["OverallQual"] * raw["GrLivArea"]).to_numpy()
    return df
```
- **第一次嘗試（v1）用過嘅特徵**：將質素等級（Ex = Excellent、Gd = Good、TA = Typical、Fa = Fair、Po = Poor）變成 5 到 1 分。
- `.map(QUALITY)`：用 dict 逐格「翻譯」，"Gd" → 4。冇呢樣嘢嘅屋（例如冇地庫）會變 NaN，再用 `fillna(0)` 當 0 分。
- `column + "Score"`：字串相加，例如 "ExterQual" + "Score" → "ExterQualScore"。
- 再加一個「質素 × 面積」嘅交互特徵，取 log1p。

```python
tried = {"Step 13 blend (kept)": rmse(blend13_oof)}
```
開一個 dict 記錄每個方法嘅 CV，第一行係我哋採用嘅 Step 13 blend，做比較基準。

```python
oof_matrix = np.column_stack(list(oof13.values()))
stacked = cross_val_predict(LinearRegression(positive=True), oof_matrix, y_log,
                            cv=KFold(n_splits=5, shuffle=True, random_state=7))
tried["stacking: weights learned from OOF predictions"] = rmse(stacked)
```
- **Stacking：** 唔用等權重，改為用一個 model 去「學」每個 model 應該佔幾多。
- `np.column_stack(...)`：將 4 個 model 嘅 OOF 預測並排，砌成一張 1460 × 4 嘅表。每行一間屋，每欄一個 model 嘅預測。
- `LinearRegression(positive=True)`：用線性回歸學 4 個權重，`positive=True` 規定權重唔可以係負數。
- **點解要用 `cross_val_predict` 同另一個 random_state（7）？** 如果用全部 1460 間學權重、再喺同一批屋度評分，就係 Part 5 講過嘅「自己改自己份卷」，分數會虛高。`cross_val_predict` 會再做一次 5-fold：每次用 4 份學權重，去估第 5 份，咁評分先公平。用唔同嘅切法（7），可以避免同第一層 CV 嘅切法完全重疊。

```python
tried["first attempt's recipe: Lasso + CatBoost 50/50"] = rmse((oof13["lasso"] + oof13["catboost"]) / 2)
```
**第一次嘗試嘅 blend 方法**：兩個 model 各佔一半（v1 用 ElasticNet，我哋用最接近嘅 Lasso）。

```python
score, _, _ = cv_evaluate(linear_pipeline(Lasso(alpha=lasso_alpha, max_iter=50000)),
                          add_quality_scores(X13, X4), y, add_quality_scores(X13_test, X4_test),
                          drop_outliers=False)
tried["Lasso alone, Step 13 features"] = rmse(oof13["lasso"])
tried["Lasso alone, + first attempt's quality scores"] = score
```
用 Lasso 比較「加唔加 v1 嘅質素特徵」。`X4` 係 Part 3 未取 log 嘅特徵，用嚟攞原始嘅質素等級同面積。

```python
xgb_runs = [oof13["xgb"]]
for seed in (7, 2026):
    _, oof_seed, _ = cv_evaluate(make_xgb(seed), X13, y, X13_test, drop_outliers=False)
    xgb_runs.append(oof_seed)
tried["XGBoost alone, seed 42"] = rmse(oof13["xgb"])
tried["XGBoost alone, average of 3 seeds"] = rmse(np.mean(xgb_runs, axis=0))
tried["Step 13 blend, but with the 3-seed XGBoost"] = rmse(np.mean(
    [oof13["lasso"], np.mean(xgb_runs, axis=0), oof13["catboost"], oof13["svr"]], axis=0))
```
- **Seed averaging：** XGBoost 有隨機成份（`subsample`、`colsample_bytree`），唔同 seed 會得出少少唔同嘅 model。用 3 個 seed（42、7、2026）各訓練一次再平均，可以減少隨機性。
- 最後一行：將 3-seed 版本嘅 XGBoost 放入 blend，睇下 blend 會唔會好啲。

```python
pd.Series(tried).round(5).to_frame("cv_rmse")
```
將 dict 變成一張表顯示。

跑出嚟會見到：

```text
                                                cv_rmse
Step 13 blend (kept)                            0.11910
stacking: weights learned from OOF predictions  0.12260
first attempt's recipe: Lasso + CatBoost 50/50  0.12026
Lasso alone, Step 13 features                   0.12559
Lasso alone, + first attempt's quality scores   0.12606
XGBoost alone, seed 42                          0.12347
XGBoost alone, average of 3 seeds               0.12326
Step 13 blend, but with the 3-seed XGBoost      0.11911
```

**點睇：**
- **Stacking 差咗（0.1226 vs 0.1191）。** 學出嚟嘅權重遷就咗訓練數據嘅雜訊，尤其係嗰兩間誤差極大嘅 outlier，所以喺新數據上反而差。同 Part 5 講「唔好調權重」嘅道理一樣。
- **第一次嘗試嘅 blend 方法（Lasso + CatBoost 各半）差咗（0.1203）。** 即係 v1 喺 Kaggle 贏，唔係因為佢嘅 blend 方法好啲。
- **第一次嘗試嘅質素特徵令 Lasso 差咗少少（0.12559 → 0.12606）。** 同 v1 自己做過嘅實驗結論一致：喺佢嘅 ablation 入面，質素特徵組亦都係冇幫助。
- **Seed averaging：** XGBoost 單獨用 3 個 seed 平均，好咗少少（0.12347 → 0.12326）；但放入 blend 之後完全冇分別（0.11910 vs 0.11911）。原因係 blend 本身已經係平均緊 4 個 model，XGBoost 嘅隨機性早已經被抵消咗。所以唔值得為咗佢令 notebook 慢多幾倍。

**結論：** 呢啲「最終手段」都冇帶來實質改善。第一次嘗試喺 Kaggle 分數好啲，最可能嘅原因係佢對 Id 2550 呢類屋估得平啲，再加上運氣，而唔係佢嘅方法整體上好啲。

---

## 8. Cell 8：更新 submission 清單

```python
summary = pd.concat([summary, pd.DataFrame([{"submission": "s7_outlier_flag", "step": "13",
                                             "cv_rmse": results.set_index("step").cv_rmse["13"]}])],
                    ignore_index=True)
summary.to_csv(OUT_DIR / "submission_summary.csv", index=False)
summary.round(5)
```
- 喺 Part 5 嘅 submission 總表加多一行 S7。
- `pd.concat([舊表, 新一行])` 將兩張表上下疊埋（→ Part 0 §3.3）。
- `results.set_index("step").cv_rmse["13"]`：喺成績表查 Step 13 嘅 CV（→ Part 0 §3.1）。

跑出嚟會見到：

```text
        submission step  cv_rmse
0     s1_ridge_raw    1  0.15534
1  s2_new_features    4  0.15121
2  s3_log_features    5  0.13239
3   s4_no_outliers   6b  0.12766
4      s5_catboost   10  0.12320
5         s6_blend   12  0.11906
6  s7_outlier_flag   13  0.11910
```

---

## 9. 交 S7 上 Kaggle：會點？

`s7_outlier_flag.csv` 同 S6 比，基本上只係 Id 2550 嘅估價由 64.3 萬改成 26.0 萬，其他屋幾乎唔變。所以 S7 嘅 Kaggle 分數**主要取決於 Id 2550 嘅真實價錢**，而呢個我哋唔知：

| 如果 Id 2550 真實價錢係 | S7 對比 S6 | 原因 |
|---|---|---|
| 同佢兩個兄弟差唔多（大約 16 至 20 萬） | **好好多**（粗略估計大約 0.116） | 26 萬比 64 萬接近真實價錢好多 |
| 大約 40 萬 | 差唔多 | 兩個估價同真實價錢嘅距離差唔多 |
| 好貴（例如 60 萬以上） | 差咗 | 26 萬估得太平 |

（「大約 0.116」係假設 leaderboard 有計呢間屋、其他屋嘅誤差不變，粗略推算出嚟。如果 Kaggle 只用部份 test 屋計分，而啱啱冇計呢間，S7 同 S6 就會差唔多。）

**我哋嘅判斷係第一種情況機會大啲**，因為 train 入面同類嘅例子 2 個中 2 個都賣得好平（16 至 18.5 萬）。留意：一般嘅 Partial sale 並唔平，train 入面 125 間 Partial sale 嘅中位數係 24.5 萬，比整體仲貴；平嘅只係「超大」嗰類。所以**呢個係根據兩個例子作出嘅判斷，唔係保證**。

**點樣交：**
- 只交 **S7 一個**，Description 寫「Step 13: keep outliers + HugePartial flag」。
- **無論結果點，都照實記錄同報告。** 唔好因為結果唔好就交第二個、第三個版本去試，咁就變成 leaderboard probing（Part 5 §1.3 講過嘅「用同一批數據揀嘢」）。

---

## 10. 報告點寫呢一段

呢個 Part 正正係講師最想睇嘅 **pathway**：遇到預期之外嘅結果，點樣分析、點樣驗證、點樣修正。建議結構：

1. **觀察：** Kaggle 同 CV 唔一致（Cell 1 嘅表）。
2. **分析：** CV 有盲點，兩間 outlier 喺同一個 fold，而佢哋貢獻咗大約 17% 嘅誤差平方（Cell 2）；test 有一間同類嘅屋（Cell 3）。
3. **假設：** Step 6 刪走 outlier，令 model 學唔到呢類屋。
4. **驗證：** 自己設計 leave-one-outlier-out 測試，誤差大幅收窄（Cell 5）。
5. **修正：** Step 13 保留 outlier，加 HugePartial 標記（Cell 6）。
6. **對照：** 其他方法（stacking、seed averaging、v1 嘅方法）都試過，冇幫助（Cell 7）。
7. **坦白嘅限制：** Step 13 係睇咗 leaderboard 先觸發嘅；我哋只交咗一次去驗證，冇反覆試 leaderboard。

**個人問題「遇到嘅困難」都可以用呢段：** 「重做之後，CV 好咗，但 Kaggle 分數反而差過第一次。我哋冇盲目試更多 model，而係做 error analysis，發現咗 CV 嘅盲點同一間特別嘅 test 屋。」

---

## 11. 講師可能會問

**Q：你哋 CV 進步咗，但 Kaggle 冇進步，點解？**
A：有兩個原因。第一，0.001 至 0.003 級數嘅差異細過抽樣運氣，同一個 model 喺唔同 fold 嘅分數可以相差 0.05。第二，我哋嘅 CV 有盲點：train 嘅兩間「超大 Partial sale」啱啱喺同一個 fold，所以 CV 測試唔到 model 處理呢類屋嘅能力，而 test 入面剛好有一間。

**Q：咩係 error analysis？你哋點做？**
A：即係唔淨係睇總分，而係睇「錯喺邊」。我哋計咗每間屋嘅 OOF 誤差，發現 2 間屋（0.14%）貢獻咗大約 17% 嘅誤差平方，再睇佢哋嘅特徵同 fold 位置，搵出 CV 嘅盲點。

**Q：Step 6 已經拎走咗 outlier，點解 Step 13 又保留返？係咪自相矛盾？**
A：唔係矛盾，係深入咗。Step 6 嘅 CV 證明「拎走」令正常屋估得準啲，但代價係 model 完全唔識呢類特殊屋。Step 13 用一個更好嘅方法：保留佢哋，加一個標記，等 model 可以分開處理。正常屋唔受影響（預測中位數只變咗 0.25%），特殊屋就估得準好多。

**Q：加 HugePartial 標記，係咪特登為 test 嗰一間屋而做？算唔算作弊？**
A：唔算作弊，因為我哋冇用 test 嘅答案，只係用咗 train 數據嘅規律（兩個例子都賣得平）同 test 嘅特徵（睇特徵係完全合規嘅）。不過我哋坦白承認：呢一步係睇到 leaderboard 結果之後先做嘅分析，所以我哋只交一次 S7 去驗證，冇反覆試。

**Q：點解 CV 分數冇因為 Step 13 而改善？咁點知佢有用？**
A：因為兩間 outlier 喺同一個 fold，標準 CV 冇辦法測試呢個做法。所以我哋另外設計咗 leave-one-outlier-out 驗證：每次用一間訓練、估另一間，log 誤差由 −1.28 / −1.49 收窄到 −0.27 / −0.64。

**Q：點解唔用 stacking？**
A：我哋試過，用 nested CV 公平咁評分，結果差過等權重 blend（0.1226 vs 0.1191）。學出嚟嘅權重會遷就訓練數據嘅雜訊。

---

## 課堂連結

- Model evaluation：cross-validation 嘅限制、validation design、overfitting to the leaderboard
- Data preprocessing：outlier handling、data cleaning（唔可能嘅數值）
- Feature engineering：indicator / flag features
- Ensemble methods：stacking
