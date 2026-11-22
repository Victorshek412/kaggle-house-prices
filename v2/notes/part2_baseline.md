# Part 2 學習筆記：Baseline 同 Cross-validation

> 呢個 Part 係成本 notebook 嘅**地基**。之後每一步都會用到呢度寫嘅 4 個工具：
> `make_linear_preprocessor`、`cv_evaluate`、`record`、`save_submission`。
> 呢度明晒，之後每一步就只係「改一樣嘢，再量一次」。

> **前置知識（唔識就先睇 Part 0）：** function 同預設值 → §1.3；`for` 同 `if` → §1.4、§1.5；numpy array 同 `np.clip`、`np.zeros` → §2；X、y、`fit`、`predict` → §4.1、§4.2；coefficient → §4.3；overfitting → §4.4；hyperparameter → §4.5；Transformer 同 Pipeline → §4.6；RMSE → §4.7。

---

## 1. 概念

### 1.1 點解要有 baseline

Baseline 係一個「最笨嘅答案」，俾你知「乜都唔做」會得幾多分。

- **Step 0** 對每間屋都估同一個價錢，即係平均價。
- **Step 1** 用最簡單嘅 Ridge，數據幾乎唔處理。

之後每一步嘅進步，都係同呢兩個比較。冇 baseline，你講「我得 0.13」就冇人知係好定差。

### 1.2 Cross-validation（5-fold）

```
1460 間屋 → 隨機分成 5 份（fold），每份大約 292 間

第 1 輪：  [驗] [訓] [訓] [訓] [訓]   → 用 4 份訓練，估第 1 份
第 2 輪：  [訓] [驗] [訓] [訓] [訓]   → 估第 2 份
…
第 5 輪：  [訓] [訓] [訓] [訓] [驗]   → 估第 5 份
```

做完 5 輪，**每間屋都被一個「從未見過佢」嘅 model 估過一次**，呢啲預測叫 **out-of-fold（OOF）預測**。將全部 OOF 預測同真實價錢比較，計出嚟嘅誤差就係 **CV 分數**。

**點解唔用 test 計分？** test 冇答案。Kaggle 每日亦有 submission 上限，唔可以靠佢試嘢。

**點解唔直接用 train 計分？** model 已經見過答案，好似考試前睇咗試卷，分數一定好過真實水平。

**點解每一步都要用同一個切法（`random_state=42`）？** 唔同切法本身就會令分數有少少差異。用同一個切法，兩個 step 之間嘅分別先可以肯定係嚟自你改嘅嘢，而唔係運氣。

### 1.3 Data leakage（數據洩漏）

有啲預處理會**從數據度學嘢**：

| 預處理 | 學咗咩 |
|---|---|
| 用 median 填缺失值 | 學咗「中位數係幾多」 |
| StandardScaler 標準化 | 學咗平均值同標準差 |
| One-hot encoding | 學咗「有邊幾個類別」 |

如果喺 5-fold 之前就用晒 1460 間屋做呢啲嘢，驗證嗰份嘅屋其實已經「影響咗」訓練，CV 分數就會比真實好。呢個就叫 **leakage**。

**解決方法：** 將預處理同 model 一齊放入 `Pipeline`。每個 fold fit 嗰陣，佢只會從嗰個 fold 嘅訓練數據學。

> 一句講晒：**任何「睇成欄數據先做到」嘅步驟，都要放入 Pipeline。**

### 1.4 預處理三件事

1. **Impute（填缺失）**
   - 數字欄用 median。點解唔用 mean？因為 median 唔會被極端值拉走。
   - 文字欄填 `"Missing"`，將「缺失」當成一個類別。
2. **Scale（標準化）**：將每個數字欄變成平均 0、標準差 1。Ridge 嘅 penalty 會對每個係數一視同仁。如果唔標準化，以「平方呎」計嘅欄（數值幾千）同以「間」計嘅欄（數值 1–4）就唔公平。
3. **One-hot encoding**：model 只識數字，所以要將文字欄轉做 0/1 欄。例如 `Street = Pave / Grvl` 會變成 `Street_Pave` 同 `Street_Grvl` 兩欄。79 欄做完之後大約變成 270 欄。

### 1.5 Ridge regression

普通線性回歸只係減低誤差：

```
盡量細：Σ (真實 − 估計)²
```

Ridge 喺後面加多一個 penalty：

```
盡量細：Σ (真實 − 估計)²  +  alpha × Σ (係數)²
```

- **直覺：** 270 欄但只有 1460 行，普通回歸好容易「死記」訓練數據（overfit），某啲係數會變得好大。penalty 就係話「係數越大越要罰」，逼 model 簡單啲。
- `alpha` 越大，罰得越重。0 代表等於普通回歸；太大就乜都學唔到（underfit）。
- Part 3 會用 CV 揀 alpha。

---

## 2. 逐行解釋

### Cell 1：Import 同工具

```python
from sklearn.base import clone
```
`clone(model)` 會整一個**全新、未訓練過**嘅 model 副本，每個 fold 都要由零開始。冇咗佢，第 2 輪會喺第 1 輪訓練過嘅 model 上面繼續訓練，即係 leakage。

```python
from sklearn.compose import ColumnTransformer, make_column_selector
```
- `ColumnTransformer`：唔同類型嘅欄做唔同嘅預處理（數字一套，文字一套）。
- `make_column_selector`：按資料類型自動揀欄，費事人手打 79 個欄名。之後 Part 3 加新欄，佢都會自動包埋。

其餘 import 都係下面會用到嘅工具：DummyRegressor、SimpleImputer、Ridge、KFold、Pipeline、OneHotEncoder、StandardScaler。

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```
- 5 份。
- `shuffle=True`：分之前先洗牌，費事原本數據嘅排列次序影響結果。
- `random_state=42`：固定個隨機種子，每次都切得一樣。

```python
def make_linear_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
```
數字欄嘅流水線：先填 median，再標準化。**次序重要**：要先填，因為 StandardScaler 唔識處理缺失值。

```python
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
```
- 文字欄：缺失填 `"Missing"`，然後做 one-hot。
- `handle_unknown="ignore"`：如果 test 出現 training 從未見過嘅類別，就將佢所有 one-hot 欄設做 0，而唔係報錯。
- 冇咗呢個設定會點：Part 1 見到有啲類別只喺 test 出現，冇咗呢句 notebook 就會爆。

```python
    return ColumnTransformer([
        ("num", numeric, make_column_selector(dtype_include="number")),
        ("cat", categorical, make_column_selector(dtype_exclude="number")),
    ])
```
數字欄行數字流水線，其他欄（文字）行文字流水線。

```python
def cv_evaluate(model, X, y, X_test, target="log", drop_outliers=False):
```
成本 notebook 最重要嘅 function。輸入 model 同數據，輸出三樣嘢：CV 分數、OOF 預測、test 預測。

- `target`：決定學 log 價錢定原價。
- `drop_outliers`：Step 6 先會用到。

```python
    y = np.asarray(y, dtype=float)
    y_log = np.log(y)
    oof_log = np.zeros(len(X))
    test_log = np.zeros(len(X_test))
```
- 先準備兩個全部係 0 嘅 array：一個放 1460 個 OOF 預測，一個放 1459 個 test 預測。
- `np.asarray` 將 pandas Series 轉做 numpy array，之後用位置索引（`y[train_idx]`）先唔會出錯。

```python
    for train_idx, valid_idx in kf.split(X):
```
- 行 5 次，每次 `kf.split` 會俾出兩組**行號**：一組係訓練嘅行，一組係驗證嘅行。
- 一次拆兩個值落 `train_idx` 同 `valid_idx`（unpacking → Part 0 §1.4）。
- `X.iloc[train_idx]` 嘅意思係「用位置揀出呢啲行」（→ Part 0 §3.2）。

```python
        if drop_outliers:
            train_idx = train_idx[~outlier.values[train_idx]]
```
- `outlier` 係 Part 1.4 用**原始數據**計出嚟嘅 True/False mask。
- `outlier.values[train_idx]` 會攞出訓練行各自係咪 outlier；`~` 將 True/False 反轉（→ Part 0 §2.3），即係「唔係 outlier 嘅先留低」。
- 只喺**訓練行**入面拎走 outlier。
- **點解唔喺呢度直接寫 `X.GrLivArea > 4000`？** 因為 Part 3 Step 5 會將 GrLivArea 取 log，log 之後最大都係 8.6 左右，「> 4000」就永遠唔會成立。第一版 code 真係中咗呢個伏，Step 6 完全冇效果。詳情見 Part 3 筆記。
- `valid_idx` 完全冇郁，outlier 照樣要被估、照樣計分。

```python
        fold_model = clone(model)
```
每輪一個全新 model。`fold_model.fit(...)` 會令 pipeline 逐步學嘢：先學 median 同 scaling，再學 one-hot 類別，最後學 Ridge 嘅 coefficient（→ Part 0 §4.6）。

```python
        if target == "log":
            fold_model.fit(X.iloc[train_idx], y_log[train_idx])
            oof_log[valid_idx] = fold_model.predict(X.iloc[valid_idx])
            test_log += fold_model.predict(X_test) / kf.n_splits
```
- 用訓練行學 log 價錢。
- 估驗證行，放入 `oof_log` 對應嘅位置。
- 同時估 test，每輪加 1/5，即係 **5 個 model 嘅平均**。平均多個 model 嘅結果，比只信一個 model 穩定。

```python
        else:
            fold_model.fit(X.iloc[train_idx], y[train_idx])
            floor = y[train_idx].min() / 2
            oof_log[valid_idx] = np.log(np.clip(fold_model.predict(X.iloc[valid_idx]), floor, None))
            test_log += np.log(np.clip(fold_model.predict(X_test), floor, None)) / kf.n_splits
```
- 學原價（Step 0、1 用）。
- 線性 model 估原價，理論上可以估出負數，但 `log(負數)` 係冇定義嘅，會爆。
- `np.clip(…, floor, None)` 嘅意思係「低過 floor 就當 floor」。floor 設做訓練數據最平價錢嘅一半。
- 今次實際冇一個預測撞到 floor（最低嘅預測都有 27,000 蚊），但留住佢做保險。
- 估完再轉做 log，因為計分要用 log。

```python
    cv_rmse = np.sqrt(np.mean((y_log - oof_log) ** 2))
    return cv_rmse, oof_log, test_log
```
就係 Kaggle 嗰條公式：差 → 平方 → 平均 → 開方。

```python
results = pd.DataFrame(columns=["step", "description", "cv_rmse"])

def record(step, description, cv_rmse):
    global results
```
- `results` 係成本 notebook 共用嘅成績表。
- `global` 係話俾 Python 聽，喺 function 入面改嘅係外面嗰個 `results`，唔係開一個新嘅。

```python
    if len(results):
        change = cv_rmse - results.cv_rmse.iloc[-1]
```
同上一行（`iloc[-1]` 即係最後一行）比較。**負數代表進步**，因為誤差細咗。

```python
    results.to_csv(OUT_DIR / "cv_results.csv", index=False)
```
每次都存檔。最後畫「進步樓梯」圖同寫報告，都係用呢個檔案。

```python
def save_submission(name, test_log):
    submission = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp(test_log)})
```
`np.exp` 將 log 變返做價錢，因為 Kaggle 要嘅係價錢。

```python
    assert len(submission) == 1459 and submission.Id.tolist() == sample.Id.tolist()
    assert np.isfinite(submission.SalePrice).all() and (submission.SalePrice > 0).all()
```
- `assert` 嘅意思係「如果唔係咁，即刻停低報錯」。
- 呢兩句喺交之前檢查三樣嘢：行數啱、Id 次序同範本一樣、冇 NaN、冇無限大、冇負數。
- 寧願喺 notebook 爆，都好過浪費一次 Kaggle submission。

### Cell 2：Step 0

```python
X = train.drop(columns=["Id", "SalePrice"])
y = train.SalePrice
X_test = test.drop(columns="Id")
```
- `X` 係特徵，`y` 係答案。
- Id 只係編號，冇預測能力，如果放入去 model 可能會學到假規律。

```python
score, _, _ = cv_evaluate(DummyRegressor(strategy="mean"), X, y, X_test, target="raw")
```
- `DummyRegressor` 會完全無視 X，淨係估訓練數據嘅平均價。
- `_` 係 Python 嘅慣例，代表「呢個回傳值我唔要」。

### Cell 3：Step 1

```python
ridge_raw = Pipeline([("prep", make_linear_preprocessor()), ("model", Ridge(alpha=10))])
```
預處理同 Ridge 串埋一齊，`cv_evaluate` 每個 fold 會成條 pipeline 一齊 fit，所以冇 leakage。`alpha=10` 暫時係隨便揀嘅中間值。

```python
score, oof_step1, test_step1 = cv_evaluate(ridge_raw, X, y, X_test, target="raw")
record("1", "Ridge, raw price, minimal preprocessing", score)
save_submission("s1_ridge_raw", test_step1)
```
評分、記錄、出一個 submission 檔案。呢個就係你**第一次 Kaggle submission（S1）**。

---

## 3. 點睇結果

| Step | CV RMSE | 大約誤差 |
|---|---|---|
| 0 估平均價 | **0.408** | 典型誤差大約 50% |
| 1 Ridge 原價 | **0.155** | 典型誤差大約 17% |

- 點樣由 RMSE 換算做百分比：`e^0.155 − 1 ≈ 0.17`。呢個只係粗略直覺，用嚟講解就夠。
- 0.408 → 0.155 嘅進步好大，但**呢個唔應該喺報告入面當成「最大進步」**。由「乜都唔做」去到「有個 model」一定會大幅進步，冇乜研究價值。報告要講嘅「best improvement」應該喺 Part 3 揀。
- 你之後交 S1 上 Kaggle，個 leaderboard 分數應該同 0.155 差唔遠。如果相差好遠，就代表 CV 唔可靠，要搵原因。

---

## 4. 講師可能會問

**Q：點解用 5-fold，唔用 train/test split 一次過切 80/20？**
A：只切一次，分數會好受「啱啱切到邊啲屋」影響。5-fold 令每間屋都做過一次驗證，分數穩定好多，而且全部 1460 間屋都有用到。

**Q：咩係 data leakage？你哋點樣避免？**
A：即係驗證數據嘅資訊漏咗入訓練過程，令 CV 分數虛高。我哋將所有「需要從數據學嘢」嘅預處理（填 median、標準化、one-hot）放入 sklearn Pipeline，每個 fold 只用自己嘅訓練數據 fit。

**Q：Ridge 同普通線性回歸有咩分別？alpha 係咩？**
A：Ridge 多咗一項 penalty，會懲罰大嘅係數，防止喺特徵多、數據少嘅情況下 overfit。alpha 控制懲罰幾重，我哋喺 Part 3 用 CV 揀。

**Q：點解 test 嘅預測係 5 個 model 平均？**
A：5 個 fold 有 5 個稍為唔同嘅 model。平均佢哋嘅預測可以抵消個別 model 嘅隨機誤差，結果更穩定，亦唔使另外再用全部數據 train 多一次。

**Q：點解標準化前要先填缺失值？**
A：StandardScaler 要計平均同標準差，有缺失值計唔到。而且 pipeline 係按次序執行嘅。

**Q：`handle_unknown="ignore"` 有咩用？**
A：test 可能出現 training 從未見過嘅類別。冇呢個設定，OneHotEncoder 會報錯；有咗佢，嗰個類別就會變成全部 0，等同「唔屬於任何已知類別」。

---

## 課堂連結

- Model evaluation：cross-validation、overfitting
- Data preprocessing：normalization、missing values、categorical encoding
- Regression：linear regression、regularization
