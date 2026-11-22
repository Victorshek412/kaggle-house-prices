# Part 2 學習筆記：Baseline 同 Cross-validation

```python setup
import sys, numpy as np, pandas as pd
sys.path.insert(0, "tools")
from note_helpers import load_data, load_defs
train, test = load_data()

from pathlib import Path
import tempfile
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold
OUT_DIR = Path(tempfile.mkdtemp())
results = pd.DataFrame(columns=["step", "description", "cv_rmse"])
load_defs("part2_baseline.py", ["make_linear_preprocessor", "record"], globals())
```


> 呢個 Part 係成個 notebook 嘅**地基**。之後每一步都會用到呢度寫嘅 4 個工具：
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

**用 10 間屋睇清楚 5-fold 點切：**

```python run
houses = np.arange(10)                    # 10 間屋，編號 0 至 9
kf_demo = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, valid_idx) in enumerate(kf_demo.split(houses), start=1):
    print(f"fold {fold}: learn from {train_idx.tolist()}  ->  predict {valid_idx.tolist()}")
```

**點睇：** 每一輪用 8 間屋學、估另外 2 間。5 輪之後，0 至 9 每一間都**啱啱好被估過一次**，而且估佢嗰陣，佢唔喺學習嗰 8 間入面。好似輪流做考官：每個人都會考一次，但考自己嗰次唔可以睇答案。

```python run
oof = np.zeros(10)
for fold, (train_idx, valid_idx) in enumerate(kf_demo.split(houses), start=1):
    oof[valid_idx] = fold                 # 用 fold 編號代替「預測」，睇下邊格由邊輪填
print(oof)
```

**點睇：** 呢個就係 `cv_evaluate` 入面 `oof_log[valid_idx] = ...` 做緊嘅嘢：每一輪只填佢負責嗰兩格，5 輪之後 10 格全部填滿，冇一格漏、冇一格重複。真正嘅 notebook 填嘅係預測價錢，唔係 fold 編號。

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

**用真數據睇 leakage 係咩：**

```python run
lot = train["LotFrontage"]
train_idx, valid_idx = next(KFold(n_splits=5, shuffle=True, random_state=42).split(lot))
print("median from all 1460 houses:          ", lot.median())
print("median from the fold's 1168 train rows:", lot.iloc[train_idx].median())
```

**點睇：** 兩個 median 唔同：用全部屋計，有份計嘅包括咗驗證嗰 292 間。即係驗證屋嘅資料「漏」咗入訓練用嘅數字度，呢個就係 leakage。差距今次好細，但原則好重要；有時差距會大到令 CV 分數明顯虛高。

> 一句講晒：**任何「睇成欄數據先做到」嘅步驟，都要放入 Pipeline。**

### 1.4 預處理三件事

1. **Impute（填缺失）**
   - 數字欄用 median。點解唔用 mean？因為 median 唔會被極端值拉走。
   - 文字欄填 `"Missing"`，將「缺失」當成一個類別。
2. **Scale（標準化）**：將每個數字欄變成平均 0、標準差 1（公式同例子 → Part 0 §4.6）。
   - **點解 Ridge 需要？** 舉個例：面積每多 1 呎可能只值 50 蚊（係數 50），睡房每多 1 間值 10,000 蚊（係數 10000）。
   - Ridge 嘅 penalty 罰嘅係「係數²」，所以睡房嘅大係數會被罰到幾乎用唔到，只係因為佢嘅單位（間）細，唔係因為佢唔重要。
   - 標準化之後，兩欄都喺同一把尺上，係數大細先反映真正嘅重要性，penalty 先公平。
3. **One-hot encoding**：model 只識數字，所以要將文字欄轉做 0/1 欄。例如 `Street = Pave / Grvl` 會變成 `Street_Pave` 同 `Street_Grvl` 兩欄。79 欄做完之後大約變成 **300 欄**（36 個數字欄 + 大約 268 個 one-hot 欄）。去到 Part 4 會變成 324 欄，因為 Part 3 加咗新特徵、`MSSubClass` 變咗類別、多咗 NoGarage 呢類標籤。

**用真數據睇預處理做咗咩：**

```python run
X_demo = train.drop(columns=["Id", "SalePrice"])
prep = make_linear_preprocessor()
matrix = prep.fit_transform(X_demo)
print("before:", X_demo.shape, "  after:", matrix.shape)
names = prep.get_feature_names_out()
print([n for n in names if "Street" in n])
column = list(names).index("num__GrLivArea")
print("house 0 GrLivArea:", X_demo["GrLivArea"].iloc[0], "->", round(matrix[0, column], 3))
```

**點睇：**
- 79 欄變成 304 欄：文字欄被 one-hot 拆開。
- 例如 Street 變成兩欄：`cat__Street_Grvl` 同 `cat__Street_Pave`（`cat__` 代表嚟自文字欄嗰條流水線）。
- 第 0 間屋嘅居住面積 1710 呎，標準化之後變成大約 0.37，即係「比平均大少少」（→ Part 0 §4.6）。

### 1.5 Ridge regression

普通線性回歸只係減低誤差：

```
盡量細：Σ (真實 − 估計)²
```

Ridge 喺後面加多一個 penalty：

```
盡量細：Σ (真實 − 估計)²  +  alpha × Σ (係數)²
```

- **直覺：** 大約 300 欄但只有 1460 行，普通回歸好容易「死記」訓練數據（overfit），某啲係數會變得好大。penalty 就係話「係數越大越要罰」，逼 model 簡單啲。
- `alpha` 越大，罰得越重。0 代表等於普通回歸；太大就乜都學唔到（underfit）。
- Part 3 會用 CV 揀 alpha。

---

## 2. 逐行解釋

### Cell 1：Import 同工具

```python
from sklearn.base import clone
```
`clone(model)` 會整一個**全新、未訓練過**嘅 model 副本，設定一樣，但乜都未學。

- **點解要用？** 確保每個 fold 用一個乾淨、獨立嘅 model，亦唔會改動外面傳入嚟嘅 `model`。
- 老實講，我哋用嘅 model（Ridge、Pipeline、XGBoost、CatBoost）每次 `fit` 本身都會由零開始，所以冇 `clone` 分數都會一樣。呢度主要係**保險同清晰**：有啲 model（例如設咗 `warm_start=True`）會喺上一次嘅結果上面繼續學，嗰陣冇 `clone` 就會出事。

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
- `handle_unknown="ignore"`：如果預測時出現訓練時從未見過嘅類別，就將佢所有 one-hot 欄設做 0，而唔係報錯。
- **幾時會發生？** 有兩種情況：
  1. 喺 CV 入面，某個罕見類別可能只出現喺驗證嗰份，訓練嗰 4 份從未見過。
  2. Part 3 Step 3 將 `MSSubClass` 變成類別之後，代號 `"150"` 只係喺 test 出現。
- 冇咗呢個設定，就會喺上面兩種情況報錯，notebook 停低。

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
成個 notebook 最重要嘅 function。輸入 model 同數據，輸出三樣嘢：CV 分數、OOF 預測、test 預測。

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

**實際切出嚟係咁**（每個 fold 嘅行數，同驗證嗰份頭 3 間屋嘅 Id）：

```text
fold 1: train 1168 rows, valid 292 rows, first valid Ids [16, 24, 30]
fold 2: train 1168 rows, valid 292 rows, first valid Ids [4, 6, 11]
fold 3: train 1168 rows, valid 292 rows, first valid Ids [1, 3, 7]
fold 4: train 1168 rows, valid 292 rows, first valid Ids [5, 12, 17]
fold 5: train 1168 rows, valid 292 rows, first valid Ids [2, 9, 14]
```

1168 + 292 = 1460。每間屋只會喺其中一個 fold 做驗證，例如 Id 1 喺 fold 3。因為 `shuffle=True`，驗證屋係打亂咗嘅，唔係頭 292 間。
- 行 5 次，每次 `kf.split` 會俾出兩組**行號**：一組係訓練嘅行，一組係驗證嘅行。
- 一次拆兩個值落 `train_idx` 同 `valid_idx`（unpacking → Part 0 §1.4）。
- `X.iloc[train_idx]` 嘅意思係「用位置揀出呢啲行」（→ Part 0 §3.2）。

```python
        if drop_outliers:
            train_idx = train_idx[~outlier.values[train_idx]]
```
- `outlier` 係 Part 1（Cell 5）用**原始數據**計出嚟嘅 True/False mask。
- `outlier.values[train_idx]` 會攞出訓練行各自係咪 outlier；`~` 將 True/False 反轉（→ Part 0 §2.3），即係「唔係 outlier 嘅先留低」。
- 只喺**訓練行**入面拎走 outlier。
- **點解唔喺呢度直接寫 `X.GrLivArea > 4000`？** 因為 Part 3 Step 5 會將 GrLivArea 取 log，log 之後最大都係 8.6 左右，「> 4000」就永遠唔會成立。第一版 code 真係中咗呢個伏，Step 6 完全冇效果。詳情見 Part 3 筆記 §6.3。
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
- 同時估 test，每輪加 1/5（`kf.n_splits` 就係 5，即係 fold 嘅數目），5 輪加完就係 **5 個 model 嘅平均**。平均多個 model 嘅結果，比只信一個 model 穩定。

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
- floor 大約係 17,450 蚊（最平嘅訓練屋 34,900 蚊嘅一半）。今次實際冇一個預測低過 floor，但留住佢做保險。
- 估完再轉做 log，因為計分要用 log。

```python
    cv_rmse = np.sqrt(np.mean((y_log - oof_log) ** 2))
    return cv_rmse, oof_log, test_log
```
就係 Kaggle 嗰條公式：差 → 平方 → 平均 → 開方。

```python
results = pd.DataFrame(columns=["step", "description", "cv_rmse"])
```
開一張空嘅成績表，有 3 欄：step 編號、描述、CV 分數。成個 notebook 都會共用佢。

```python
def record(step, description, cv_rmse):
    global results
    step = str(step)
    row = pd.DataFrame([{"step": step, "description": description, "cv_rmse": cv_rmse}])
```
- 三個引號嘅 docstring 喺呢度省略咗（→ Part 0 §1.3）。
- `global` 係話俾 Python 聽，喺 function 入面改嘅係外面嗰個 `results`，唔係開一個新嘅。
- `str(step)`：將 step 變做文字。因為有 "6b"、"7b" 呢類編號，全部用文字先一致。
- `row`：用一個 dict 砌一張得**一行**嘅小表（dict 外面加 `[ ]`，即係「一個 dict = 一行」）。

```python
    if step in results.step.tolist():  # re-running a cell: replace its row, don't duplicate it
        results.loc[results.step == step, ["description", "cv_rmse"]] = [description, cv_rmse]
    else:
        results = row if results.empty else pd.concat([results, row], ignore_index=True)
```
- **處理重跑：** 如果呢個 step 已經記錄過（例如你喺 Kaggle 重跑咗某一格），就**覆蓋**舊嗰行，唔好重複加多一行。冇咗呢個處理，成績表會有兩行 step 5，Part 5 嘅圖同總表就會亂。
- 未記錄過就加入新行：
  - `results.empty`：張表係咪仲未有任何行。
  - 係嘅話，新表就係 `row` 本身；
  - 否則用 `pd.concat` 將 `row` 疊喺舊表下面。`ignore_index=True` 即係重新編行號 0, 1, 2…。
  - 呢句係一行 if（→ Part 0 §1.8）。

```python
    position = results.step.tolist().index(step)
    if position > 0:
        change = cv_rmse - results.cv_rmse.iloc[position - 1]
        print(f"Step {step}: CV RMSE {cv_rmse:.5f} ({change:+.5f} vs previous step)")
    else:
        print(f"Step {step}: CV RMSE {cv_rmse:.5f}")
```
- `position`：呢個 step 喺表入面排第幾（由 0 數起）。`.index(step)` 係 list 嘅 method，回傳某個值嘅位置。
- 唔係第一行嘅話，就同**佢上一行**比較，計出變化 `change`。**負數代表進步**，因為誤差細咗。
- `{change:+.5f}`：5 個小數位，而且一定顯示正負號（→ Part 0 §1.7），所以會見到 `+0.00014` 或者 `-0.01882`。
- 第一行（step 0）冇上一行可以比，所以只印分數。

```python
    results.to_csv(OUT_DIR / "cv_results.csv", index=False)
```
每次都存檔。最後畫「進步樓梯」圖同寫報告，都係用呢個檔案。`index=False` 即係唔好將行號寫入檔案。

**試吓用 `record`，包括「重跑同一步」：**

```python run
record("0", "Predict the mean price", 0.408)
record("1", "Ridge, raw price", 0.155)
record("1", "Ridge, raw price (cell run again)", 0.155)
results
```

**點睇：** 頭兩次各加一行。第三次 step 1 已經存在，所以佢**覆蓋**咗舊嗰行，而唔係加多一行；張表仍然係 2 行。呢個就係為咗你喺 Kaggle 重跑某一格都唔會亂。

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
- 呢兩句喺交之前檢查以下幾樣嘢：行數啱（1459）、Id 次序同範本一樣、冇 NaN、冇無限大、冇負數或者 0。
- 寧願喺 notebook 爆，都好過浪費一次 Kaggle submission。

```python
    (OUT_DIR / "submissions").mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUT_DIR / "submissions" / f"{name}.csv", index=False)
    print("saved", name + ".csv")
```
- 開 `submissions` folder（已經有就唔理）。
- 存成 CSV，檔名係 `name` 加 `.csv`。`index=False` 好重要：冇咗佢，CSV 會多咗一欄行號，Kaggle 就會唔收。

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
- **點解用 `target="raw"`？** 「估平均價」就係原價嘅平均（例如 18 萬），同 Step 1 一樣喺原價尺度訓練，兩個先可以公平比較。
- `_` 係 Python 嘅慣例，代表「呢個回傳值我唔要」。

跑出嚟會見到：

```text
Step 0: CV RMSE 0.40776
```

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

跑出嚟會見到：

```text
Step 1: CV RMSE 0.15534 (-0.25243 vs previous step)
saved s1_ridge_raw.csv
```

第一行係 `record` 印嘅：分數 0.15534，同上一步（Step 0）比跌咗 0.25243。第二行係 `save_submission` 印嘅。Part 2 最後一格 `results` 會顯示成績表：

```text
  step                              description   cv_rmse
0    0                   Predict the mean price  0.407762
1    1  Ridge, raw price, minimal preprocessing  0.155335
```
評分、記錄、出一個 submission 檔案。呢個就係你**第一次 Kaggle submission（S1）**。

---

## 3. 點睇結果

| Step | CV RMSE | 大約誤差 |
|---|---|---|
| 0 估平均價 | **0.408** | 典型誤差大約 50%（notebook 寫「about 50%」） |
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
A：預測時可能遇到訓練時從未見過嘅類別：喺 CV 入面，罕見類別可能只喺驗證 fold 出現；而 MSSubClass 代號 150 只喺 test 出現。冇呢個設定，OneHotEncoder 會報錯；有咗佢，嗰個類別就會變成全部 0，等同「唔屬於任何已知類別」。

**Q：點解要用 `clone`？**
A：確保每個 fold 用一個全新、獨立嘅 model。我哋用嘅 model 每次 fit 本身都會重新開始，所以 `clone` 主要係保險；但遇到會「接住上次繼續學」嘅 model（例如 warm_start），冇 `clone` 就會出錯。

---

## 課堂連結

- Model evaluation：cross-validation、overfitting
- Data preprocessing：normalization、missing values、categorical encoding
- Regression：linear regression、regularization
