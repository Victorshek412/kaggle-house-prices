# Part 3 學習筆記：Feature engineering（一步改一樣嘢）

> **前置知識（唔識就先睇 Part 0）：** function、`return`、預設值 → §1.3；for 迴圈同 unpacking → §1.4；`.loc` 寫入、`.fillna`、`.astype`、`.copy()` → §3.2–3.4；coefficient 同「線性 model 只識加」→ §4.3；hyperparameter 同 tuning → §4.5；log、skew、log1p → §5。
>
> **同 Part 2 嘅關係：** 呢個 Part 會重複用 Part 2 寫好嘅 `cv_evaluate`、`record`、`save_submission`、`make_linear_preprocessor`。佢哋做咩，可以返去 Part 2 筆記睇。

---

## 0. 成個 Part 嘅思路同結果總覽

### 0.1 咩係 feature engineering

**用你對問題嘅理解，將原始資料改造成 model 更容易用嘅形式。**

model 唔識睇 `data_description.txt`，唔知 NA 代表「冇車房」，亦唔知樓價同面積係「按比例」咁變。呢啲知識要由我哋寫入數據度。

### 0.2 點解「一步只改一樣嘢」

科學實驗嘅原則係**控制變因**。

- 如果一次過改 3 樣嘢，分數好咗，你就唔知係邊樣嘢嘅功勞。
- 報告其中一條必答題係「邊項 feature engineering 最有用？點解？」。只有一步改一樣，先答得到呢條問題。

所以每一步都係：**同一個 model（Ridge alpha=10）、同一個 5-fold 切法，只係改一樣嘢**，然後同上一步比較。

### 0.3 結果總覽

| Step | 改咗咩 | CV RMSE | 同上一步比較 | 相對進步 |
|---|---|---|---|---|
| 1 | （Part 2 起點）Ridge，用原價 | 0.15534 | | |
| 2 | 改為預測 log 價錢 | 0.15206 | −0.00327 | 2.1% |
| 3 | 缺失值按意思填；MSSubClass 改做類別 | 0.15106 | −0.00100 | 0.7% |
| 4 | 新特徵（總面積、浴室數、樓齡等） | 0.15121 | +0.00014 | ≈ 0（差少少） |
| 5 | **偏斜面積取 log1p** | **0.13239** | **−0.01882** | **12.4%** |
| 6 | 訓練時拎走 2 個 outlier | 0.12766 | −0.00473 | 3.6% |
| 6b | 調 alpha（試 1、3、10、30、100） | 0.12766 | 0 | 10 本身已經最好 |

**由 Step 1 到 Step 6，誤差總共跌咗 17.8%（0.155 → 0.128）。**

「相對進步」嘅計法：例如 Step 5 係 0.01882 ÷ 0.15121 = 12.4%。

---

## 1. 開頭嗰段 markdown：點解有啲嘢可以喺 CV 之前做

呢個係成個 Part 最重要嘅概念。佢解釋咗點解 `fill_missing_meaning` 呢類 function 可以直接套落 train 同 test，而 median 填缺失就一定要放入 Pipeline。

| 類型 | 例子 | 會唔會 leak？ | 放邊度 |
|---|---|---|---|
| **逐行（row-wise）** | 「呢間屋 GarageArea = 0，所以 GarageType 填 NoGarage」；「TotalSF = 呢間屋嘅地面面積 + 呢間屋嘅地庫面積」 | **唔會**。每間屋只睇自己嗰行，同其他屋無關 | 可以喺 CV 之前做 |
| **睇成欄（column statistics）** | median、平均、標準差、「有邊幾個類別」 | **會**。驗證嘅屋會影響計出嚟嘅數 | 一定要放入 Pipeline |

**用一個例子去驗證：** 假設你喺 CV 之前用全部 1460 間屋計 LotFrontage 嘅 median（例如 69）。咁每個 fold 嘅驗證屋都有份影響呢個 69，佢哋就唔再係「完全未見過」。

相反，`TotalSF = GrLivArea + TotalBsmtSF` 只用到自己嗰間屋嘅兩個數，其他屋點樣都影響唔到。

**一個刻意嘅例外：** Step 5 揀「邊啲欄要取 log」時，係睇全部 train 嘅 skew 去決定。呢個違反咗上面嘅規則，但影響極細，原因喺 §5.2 解釋。notebook 開頭亦都講明咗呢點，講師問嘅話可以坦白答。

---

## 2. Step 2：預測 log(SalePrice)

### 2.1 概念

Part 1 見到 SalePrice 右偏（skew 1.88），而 Kaggle 係用 log 計分。所以合理嘅做法係直接學 log 價錢：

- **訓練嘅目標同評分嘅尺度一致。** 用原價訓練，Ridge 會盡量減低「金額」誤差，貴樓錯 5 萬會比平樓錯 1 萬罰得重好多。但 Kaggle 只理「百分比」誤差（→ Part 0 §5.2）。用 log 價錢訓練，兩者就一致。
- **分佈對稱咗。** 幾間極貴嘅屋唔會再拉走成條線。

### 2.2 逐行解釋

```python
ridge = Pipeline([("prep", make_linear_preprocessor()), ("model", Ridge(alpha=10))])
```
- 同 Part 2 Step 1 一模一樣嘅 model：預處理 + Ridge(alpha=10)。
- 改個名叫 `ridge`，因為之後 Step 3–6 都會用返佢。

```python
score, _, _ = cv_evaluate(ridge, X, y, X_test, target="log")
```
- 唯一嘅改變係 `target="log"`。
- 喺 `cv_evaluate` 入面，佢會行 `if target == "log":` 嗰條路（Part 2 筆記有逐行解釋）：用 `y_log` 訓練，預測出嚟本身已經係 log，所以唔使 clip。
- 回傳 3 個值，但只要第一個（分數），其餘用 `_` 丟棄（→ Part 0 §1.7）。

```python
record("2", "Predict log(SalePrice)", score)
```
加入成績表，並印出同上一步嘅差。

### 2.3 點睇結果：0.15534 → 0.15206（−2.1%）

**進步細過預期，點解？** 因為 X 入面嘅面積仍然係原數，而**樓價同面積嘅真實關係係「按比例」**：大一倍嘅屋大約貴一倍，唔係貴一個固定金額。用數學寫出嚟：

```
價錢 ≈ c × 面積    ⇒    log(價錢) ≈ log(c) + log(面積)
```

- 即係話，**「log 價錢」同「log 面積」之間先係一條直線**。
- Step 2 只係將一邊（價錢）取 log，另一邊（面積）仲係原數，所以條線仲係唔夠直。
- 等到 Step 5 將面積都取 log，兩邊就夾埋。**Step 2 同 Step 5 係一對**，呢點報告值得寫。

---

## 3. Step 3：缺失值按意思填 + MSSubClass 改做類別

### 3.1 概念

**第一部份：NA 嘅意思。** Part 1 §1.4 講過，好多 NA 其實代表「冇呢樣嘢」。但點樣肯定一個 NA 係「冇」，而唔係「漏咗記錄」？**要搵另一欄去證實。**

| 要填嘅欄 | 用邊欄證實 | 例子 |
|---|---|---|
| 5 個地庫欄（BsmtQual 等） | `TotalBsmtSF == 0` | 地庫面積係 0，先填 NoBasement |
| 4 個車房欄（GarageType 等） | `GarageArea == 0` | 車房面積係 0，先填 NoGarage |
| FireplaceQu | `Fireplaces == 0` | 壁爐數目係 0，先填 NoFireplace |
| PoolQC | `PoolArea == 0` | 泳池面積係 0，先填 NoPool |

如果 GarageArea 係 400（即係有車房），但 GarageType 係 NA，咁就係**真係唔知**。呢種情況**唔應該**填 NoGarage，要留返做缺失，交俾 Pipeline 當「Missing」處理。

Alley、Fence、MiscFeature 冇對應嘅面積欄可以證實，但 `data_description.txt` 明確寫咗 NA = 冇，所以直接填。

**第二部份：MSSubClass。** 佢入面係數字，例如 20、60、190，但其實係**代號**：20 代表「1946 年後嘅一層屋」，60 代表「1946 年後嘅兩層屋」。

- 如果當佢係數字，Ridge 會用一個 coefficient 乘佢，即係假設「代號越大，價錢越高（或越低）」。代號 60 唔係代號 20 嘅「三倍」，咁樣係冇意義嘅。
- 改做文字（`astype(str)`）之後，OneHotEncoder 會將每個代號變成獨立嘅 0/1 欄，每個代號有自己嘅權重。

### 3.2 逐行解釋

```python
ABSENT_RULES = [
    ("TotalBsmtSF", ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"], "NoBasement"),
    ("GarageArea", ["GarageType", "GarageFinish", "GarageQual", "GarageCond"], "NoGarage"),
    ("Fireplaces", ["FireplaceQu"], "NoFireplace"),
    ("PoolArea", ["PoolQC"], "NoPool"),
]
```
- 一個 list（→ Part 0 §1.2），入面有 4 個 tuple。每個 tuple 有 3 樣嘢：（用嚟證實嘅欄，要填嘅欄清單，填咩字）。
- **點解寫成一張表，唔逐條寫？** 規則一目了然；之後想加規則，只要加一行；亦唔會因為 copy-paste 而寫錯。
- **GarageYrBlt 點解唔喺度？** 佢係數字欄，冇車房就冇年份，冇辦法填一個「NoGarage」字。我哋留佢做缺失，Pipeline 會填 median。呢個做法唔完美，但影響好細。

```python
def fill_missing_meaning(df):
    df = df.copy()
```
- 定義 function，輸入一張表 `df`。
- `.copy()` 先整一個副本（點解要咁做 → Part 0 §3.4）。冇咗呢行，`X` 本身都會被改，之後你再跑 Step 2 就唔再係「原始數據」。

```python
    for size_column, columns, label in ABSENT_RULES:
```
逐條規則行，每次將 tuple 拆開放入 3 個名（→ Part 0 §1.4）。例如第 2 次：
- `size_column = "GarageArea"`
- `columns = ["GarageType", ...]`
- `label = "NoGarage"`

```python
        absent = df[size_column] == 0
```
一條 True/False Series：邊啲屋嘅面積（或數目）係 0。

**如果面積本身係缺失（NaN）會點？** test 有 1 間屋嘅 GarageArea 同 1 間屋嘅 TotalBsmtSF 係 NaN。`NaN == 0` 係 False（→ Part 0 §3.1），所以呢啲屋唔會被當成「冇車房 / 冇地庫」，個欄會保持缺失，之後交俾 Pipeline 當「Missing」處理。呢個正正係我哋想要嘅：唔知就係唔知，唔好亂估。

```python
        for column in columns:
            df.loc[absent & df[column].isna(), column] = label
```
- 再逐個要填嘅欄行一次。
- `absent & df[column].isna()`：「面積係 0」**同埋**「呢欄係缺失」，兩個條件都要成立（→ Part 0 §2.3）。
- `df.loc[條件, column] = label`：只喺符合條件嘅行，將呢欄改做 label（→ Part 0 §3.2）。
- **點解要加 `df[column].isna()`？** 保險起見，萬一有屋面積係 0 但個欄已經有值，就唔好覆蓋佢。

```python
    for column, label in [("Alley", "NoAlley"), ("Fence", "NoFence"), ("MiscFeature", "NoMisc")]:
        df[column] = df[column].fillna(label)
```
呢 3 欄冇得證實，直接將所有 NA 填做「冇」。`fillna` 只會改缺失嘅格，有值嘅格保持原狀。

```python
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    return df
```
- 數字轉文字，20 會變成 "20"。之後 `make_column_selector` 會自動當佢係文字欄，交俾 OneHotEncoder 處理。
- 最後交返改好嘅副本。

```python
X3, X3_test = fill_missing_meaning(X), fill_missing_meaning(X_test)
```
- 對 train 同 test 用**同一個 function**，保證處理方法完全一樣。
- 一行寫兩個賦值：左邊兩個名，右邊兩個值，按次序對應。
- 用新名（X3）而唔覆蓋 X：咁 X 仍然係原始數據，任何 step 都可以獨立重跑。

```python
score, _, _ = cv_evaluate(ridge, X3, y, X3_test, target="log")
record("3", "Missing values labelled by meaning; MSSubClass as category", score)
```
同 Step 2 一樣，只係將 X 換成 X3。

### 3.3 點睇結果：0.15206 → 0.15106（−0.7%）

**點解咁細？** 呢個發現本身值得寫入報告：**Pipeline 早已經幫你做咗大部份嘢。**

- 喺 Step 2，文字欄嘅 NA 已經被 `SimpleImputer` 填咗做 `"Missing"`，然後 one-hot 成一個「Missing」類別。
- 喺 GarageType 呢類欄，「Missing」幾乎就等於「冇車房」。所以 model 一早已經可以學到「冇車房 → 平啲」。
- Step 3 只係將「真係冇」同「唔知」分開，但後者好少（大部份 NA 都證實到係「冇」）。
- 呢一步嘅進步，有一部份應該係嚟自 MSSubClass 改做類別。但兩樣嘢係同一步做嘅，所以分唔開。如果想知，可以額外做一次只改 MSSubClass 嘅實驗。

**報告可以咁寫：** 「Step 3 將缺失值嘅意義明確化，效果細（−0.7%），因為 pipeline 嘅 constant imputation 已經隱含地將 NA 當成一個類別。呢個 step 主要提升咗數據嘅可解釋性同正確性，而唔係分數。」

---

## 4. Step 4：新特徵

### 4.1 概念

人睇一間屋，會諗「總共幾大？幾多個廁所？幾舊？」。但數據將呢啲資訊分散喺好多欄。Step 4 將佢哋**合併成人會用嘅概念**：

| 新特徵 | 公式 | 點解有用 |
|---|---|---|
| TotalSF | GrLivArea + TotalBsmtSF | 地面 + 地庫，代表總空間 |
| TotalBathrooms | FullBath + 0.5×HalfBath + BsmtFullBath + 0.5×BsmtHalfBath | 半套廁所（冇沖涼位）當半個 |
| HouseAge | YrSold − YearBuilt | 「賣嗰陣幾舊」，比「幾時起」更直接 |
| RemodelAge | YrSold − YearRemodAdd | 裝修幾新 |
| TotalPorchSF | 5 種露台面積加埋 | 戶外空間 |

**HouseAge 比 YearBuilt 好喺邊？** 一間 1990 年起嘅屋，2006 年賣時係 16 年樓，2010 年賣時係 20 年樓。買家睇嘅係樓齡，唔係建造年份。

### 4.2 逐行解釋

```python
def add_new_features(df):
    df = df.copy()
    df["TotalSF"] = df["GrLivArea"] + df["TotalBsmtSF"]
```
- `df["新欄名"] = …` 會開一個新欄。
- 兩欄相加係逐行加（向量化 → Part 0 §2.1），即係每間屋自己嘅兩個數相加。
- **如果其中一個係 NaN？** 結果都會係 NaN。test 有一間屋嘅 TotalBsmtSF 係缺失，佢嘅 TotalSF 就會係 NaN，之後交俾 Pipeline 填 median，唔會爆。

```python
    df["TotalBathrooms"] = (df["FullBath"] + 0.5 * df["HalfBath"]
                            + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"])
```
- 括號令公式可以分兩行寫。
- 0.5 係人為決定嘅權重（半套廁所當半個），係一個常見做法。

```python
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodelAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
```
- `.clip(lower=0)` 將負數變 0（→ Part 0 §3.3）。
- **點解會有負數？**
  - train 有 1 間、test 有 2 間屋嘅裝修年份遲過賣出年份，可能係賣咗先完成裝修，或者係數據錯。
  - test 仲有 1 間屋嘅建成年份遲過賣出年份，即係買樓花。
  - 樓齡係負數冇意義，所以當 0。兩條公式都要 clip，因為兩種情況都有。

```python
    porches = ["WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]
    df["TotalPorchSF"] = df[porches].sum(axis=1)
    return df
```
- `df[porches]` 攞出 5 欄（注意 `porches` 本身已經係 list，所以只有一層方括號）。
- `.sum(axis=1)` 係**每行**加總：`axis=1` 代表橫向加；`axis=0`（預設）就係每欄直向加。

```python
X4, X4_test = add_new_features(X3), add_new_features(X3_test)
score, _, test_log = cv_evaluate(ridge, X4, y, X4_test, target="log")
record("4", ...)
save_submission("s2_new_features", test_log)
```
- 喺 Step 3 嘅結果上面再加特徵，所以係**累積**嘅。
- 今次要 `test_log`，因為要出 submission **S2**。

### 4.3 點睇結果：0.15106 → 0.15121（+0.00014，即係冇分別）

**呢個唔係失敗，而係一個好嘅發現。** 原因係 Ridge 係線性 model（→ Part 0 §4.3）：

```
估計 = … + b1 × GrLivArea + b2 × TotalBsmtSF + …
```

如果 `b1 = b2`，效果就等於 `b1 × TotalSF`。即係話，**Ridge 本身已經可以自己「加埋」兩欄**，TotalSF 對佢嚟講冇新資訊。HouseAge 同理：佢等於 YrSold − YearBuilt，兩欄都已經喺 X 入面。

**咁點解仲要做？** 有兩個原因：
1. **Step 5 會用到。** log(TotalSF) ≠ log(GrLivArea) + log(TotalBsmtSF)。取 log 之後，總面積就變成一個 Ridge 自己砌唔出嚟嘅新資訊。
2. **Part 4 嘅樹 model 會用到。** 決策樹每次只睇一欄去分（例如「TotalSF > 2500？」），佢冇辦法自己將兩欄相加。所以對樹 model 嚟講，合併特徵係真正嘅新資訊。

> **報告重點：** 一個 feature 有冇用，**取決於 model**。對線性 model 冗餘嘅特徵，對樹 model 可能好有用。

---

## 5. Step 5：偏斜面積取 log1p（最大進步）

### 5.1 概念

**問題：** 面積欄同 SalePrice 一樣係右偏。例如 LotArea skew 12.2，GrLivArea 1.37。

**點解偏斜嘅特徵會令線性 model 估差？** 有兩個原因：

1. **關係唔係直線。** 上面 §2.3 講過，log 價錢同 log 面積先係直線關係。log 價錢對原面積係一條彎線，Ridge 只識畫直線，所以必然有誤差。
2. **極端值主導。** LotArea 有啲屋係 200,000 平方呎，大部份只係 10,000 左右。極大嘅值會將條線拉歪，好似 Part 1 樓價嘅情況。

**解決：** 對面積取 log1p，兩邊都喺 log 尺度，關係就拉直咗。

**用一個例子感受：**

| 屋 | 面積 | 加 100 呎 | 百分比 |
|---|---|---|---|
| 細屋 | 600 呎 | → 700 呎 | +16.7% |
| 大屋 | 4000 呎 | → 4100 呎 | +2.5% |

- 細屋加 100 呎嘅影響大好多，因為比例大好多。
- 用原數，兩者都係「+100」，Ridge 只能俾佢哋同一個影響。
- 用 log，細屋嘅 log 增加 0.154，大屋只增加 0.025。Ridge 自然就識分輕重。

**點解用 log1p，唔用 log？** 好多面積係 0，例如冇泳池、冇露台。`log(0)` 係負無限大，會爆；`log1p(0) = log(1) = 0`，啱啱好（→ Part 0 §5.4）。

**點揀邊啲欄？** skew > 0.75（常用門檻 → Part 0 §5.3）。21 個面積欄入面，20 個超過 0.75，只有 GarageArea（0.18）唔使。

### 5.2 逐行解釋

```python
AREA_COLUMNS = ["LotFrontage", "LotArea", ..., "TotalSF", "TotalPorchSF"]
```
- 人手列出所有「面積、長度同金額」類嘅欄（LotFrontage 係長度，MiscVal 係金額），包括 Step 4 新整嘅 TotalSF 同 TotalPorchSF。
- **點解唔對所有數字欄做？** 例如 OverallQual（1–10 分）、YearBuilt（年份）、HouseAge 呢類，唔係「按比例」嘅量。年份取 log 冇意義。

```python
skewness = X4[AREA_COLUMNS].skew().sort_values(ascending=False)
print(skewness.round(2).to_string())
```
- 計每欄嘅 skew，由大到細排。
- `.round(2)` 取 2 位小數；`.to_string()` 令每行都完整印出，唔會被省略。
- **印出嚟係俾人睇嘅證據**，報告可以引用：「我哋揀 skew > 0.75 嘅 20 欄」。

```python
SKEWED_COLUMNS = ["MiscVal", "PoolArea", ..., "2ndFlrSF"]
assert set(SKEWED_COLUMNS) == set(skewness[skewness > 0.75].index)
```
- 將清單**寫死**喺 code 入面。
- 跟住用 `assert` 檢查：寫死嘅清單同「skew > 0.75 嘅欄」係咪完全一樣？
  - `skewness[skewness > 0.75]` 係 boolean mask，揀出 skew 大過 0.75 嘅欄。
  - `.index` 攞欄名。
  - `set(...)` 將清單變成集合，比較時唔理次序。
- **點解要寫死，唔直接用計出嚟嘅結果？** 兩個原因：
  1. 清單一目了然，報告可以直接列出嚟。
  2. 呢個選擇只用咗特徵（面積），完全冇用到 SalePrice，所以冇洩漏答案。不過嚴格嚟講，用全部 train 計 skew 都算「睇咗驗證屋」。好在影響極細：佢只係決定「邊欄要取 log」，冇學任何同價錢有關嘅嘢。講師問嘅話，可以坦白講出呢個取捨。
- assert 喺度做保險：如果數據有變，清單同計算結果唔一致，就即刻報錯。

```python
def log_skewed(df):
    df = df.copy()
    df[SKEWED_COLUMNS] = np.log1p(df[SKEWED_COLUMNS])
    return df
```
- 一次過對 20 欄取 log1p（向量化）。NaN 取 log1p 都係 NaN，會交俾 Pipeline 填。
- **注意：** Pipeline 之後仲會做 StandardScaler，所以次序係 log1p → 填 median → 標準化。median 喺 log 尺度上計，冇問題。

```python
X5, X5_test = log_skewed(X4), log_skewed(X4_test)
score, _, test_log = cv_evaluate(ridge, X5, y, X5_test, target="log")
record("5", "log1p of skewed area columns", score)
save_submission("s3_log_features", test_log)
```
同之前一樣，套落 train 同 test，評分，出 submission **S3**。

### 5.3 點睇結果：0.15121 → 0.13239（−12.4%，全個 Part 最大）

報告必答題：「**邊項 feature engineering 最有用？點解？**」答案就係呢一步。理由可以分三層寫：

1. **數據層面：** 面積欄嚴重右偏（LotArea skew 12.2），少數極端值主導線性 model。
2. **關係層面：** 樓價同面積係比例關係。log 價錢對 log 面積先係直線，配合 Step 2 嘅 log target，兩邊尺度一致，正好符合 Ridge「只識畫直線」嘅假設。
3. **證據：** 只改呢一樣嘢，CV 就跌咗 12.4%，係所有 feature engineering step 入面最大嘅。

**額外 insight：** Step 2 單獨只有 2.1%，Step 5 有 12.4%。log target 同 log features 要**一齊用**先發揮到最大作用。另外，舊版（v1）做過另一種實驗，係由全部特徵開始逐組拎走，結論都係「log 組」最重要。兩種方法得出同一個結論，令呢個發現更可信。

---

## 6. Step 6：訓練時拎走 2 個 outlier

### 6.1 概念

Part 1 §1.5 已經解釋咗呢兩間屋（Id 524、1299）點解係 outlier：大屋賤賣，而且係 Partial sale。重點係三條規則：
1. **有理由先拎走**，唔係因為「佢令分數差」。
2. **只喺訓練時拎走。**
3. **評分時照計**，佢哋仍然喺 OOF 預測入面。

點解第 3 點咁重要？如果評分都拎走，你會將「最難估嘅兩間」排除喺考試外面，分數會虛高。真實嘅 test 入面可能都有類似嘅屋。

### 6.2 逐行解釋

```python
score, _, test_log = cv_evaluate(ridge, X5, y, X5_test, target="log", drop_outliers=True)
record("6", "Drop 2 outliers from training folds", score)
```
- 唯一改變係 `drop_outliers=True`。
- `cv_evaluate` 入面會用 Part 1 計好嘅 `outlier` mask，喺每個 fold 嘅**訓練行**拎走佢哋（Part 2 筆記有逐行解釋）。

### 6.3 一個真實嘅 bug（可以寫入個人答案「遇到嘅困難」）

第一版 code 嘅 outlier 判斷寫喺 `cv_evaluate` 入面：

```python
keep = ~((X.GrLivArea.values[train_idx] > 4000) & (y[train_idx] < 300000))
```

第一次跑出嚟，**Step 6 嘅分數同 Step 5 一模一樣**（+0.00000）。

- **偵查過程：** 拎走 2 間屋，分數完全冇變，係唔可能嘅，所以一定係冇屋被拎走。
- **原因：** Step 5 已經將 GrLivArea 取咗 log1p，最大值變咗大約 8.6。「> 4000」永遠唔成立。
- **修正：** 改用 Part 1 用**原始數據**計好嘅 `outlier` mask，咁就唔受之後任何轉換影響。
- **預防：** 加咗一個自動 test，檢查 Step 6 同 Step 5 嘅分數唔可以完全一樣。

**教訓：** 當結果「好得出奇」或者「完全冇變」，都要懷疑。特徵轉換之後，所有用到嗰欄嘅舊規則都要重新檢查。

### 6.4 點睇結果：0.13239 → 0.12766（−3.6%）

兩間屋只佔 0.14%，但效果明顯，因為 Ridge 用平方誤差：

- 喺 log 尺度上，佢哋係「極大面積 + 低價」。
- 原本條線要遷就佢哋嘅巨大誤差，就會壓低「面積越大越貴」嘅斜率。
- 拎走之後，條線回復正常，其他 1458 間屋都估得準啲。

---

## 7. 調 Ridge 嘅 alpha

### 7.1 概念

`alpha` 係 hyperparameter（→ Part 0 §4.5），之前嘅 10 係估嘅。而家特徵已經定咗，就應該揀返最適合嘅值。

- alpha 太細：penalty 弱，容易 overfit。
- alpha 太大：penalty 太強，所有係數都被壓到好細，變成 underfit。

方法：試幾個值，用同一個 CV 比較，揀最細 RMSE 嗰個。

**點解揀 1、3、10、30、100？** 呢組數大約每次 ×3，叫**對數尺度**。alpha 嘅影響係按倍數變化，1 同 2 嘅分別遠大過 100 同 101，所以要咁揀。

**格網邊緣規則：** 如果最好嘅值係最大（或最細）嗰個，真正最好嘅值可能喺範圍外面，就要擴大範圍再試。最初我只試咗 1 至 30，當時最好嘅係 30（喺邊緣），所以加咗 100。修正 outlier bug 之後，最好嘅係 10，喺中間，所以搵到真正嘅最低點。

### 7.2 逐行解釋

```python
alpha_scores = {}
```
開一個空 dict（→ Part 0 §1.2），用嚟記錄「alpha → 分數」。

```python
for alpha in [1, 3, 10, 30, 100]:
    candidate = Pipeline([("prep", make_linear_preprocessor()), ("model", Ridge(alpha=alpha))])
    alpha_scores[alpha], _, _ = cv_evaluate(candidate, X5, y, X5_test, target="log", drop_outliers=True)
    print(f"alpha={alpha:>3}: CV RMSE {alpha_scores[alpha]:.5f}")
```
- 每個 alpha 整一條新 pipeline，用完全一樣嘅設定（X5、log、拎走 outlier）評分。
- `alpha_scores[alpha], _, _ = ...` 將分數直接存入 dict 嘅對應 key。
- `{alpha:>3}` 令數字靠右、佔 3 格，印出嚟會對齊。

```python
best_alpha = min(alpha_scores, key=alpha_scores.get)
```
- 喺 dict 入面搵「分數最細」嗰個 key。
- `min(..., key=...)` 嘅意思係用 `alpha_scores.get(alpha)`（即係分數）嚟比較大細，但回傳嘅係 key（alpha）本身。
- **用例子行一次：**
  - `alpha_scores = {1: 0.1303, 3: 0.1284, 10: 0.1277, 30: 0.1280, 100: 0.1298}`
  - `min` 逐個 key 問 `alpha_scores.get(key)` 攞分數：1 → 0.1303，3 → 0.1284，10 → 0.1277……
  - 0.1277 最細，所以回傳佢嘅 key，即係 **10**。
- 留意 `.get` 後面**冇括號**：我哋係將「查分數呢個動作」交俾 `min` 自己逐個用，而唔係即刻查一次。

```python
ridge = Pipeline([("prep", make_linear_preprocessor()), ("model", Ridge(alpha=best_alpha))])
score, oof_ridge, test_log = cv_evaluate(ridge, X5, y, X5_test, target="log", drop_outliers=True)
record("6b", f"Ridge alpha tuned to {best_alpha}", score)
save_submission("s4_no_outliers", test_log)
```
- 用最好嘅 alpha 重新整 ridge，評分、記錄，出 submission **S4**。
- `oof_ridge` 會留低，Part 5 blend 嗰陣用。

```python
X_final, X_test_final = X5, X5_test
```
- 改個清楚嘅名，話俾之後嘅 Part 知：「呢個係最終特徵」。
- Part 4 嘅線性 model 同 SVR 會用佢。

### 7.3 點睇結果

| alpha | 1 | 3 | **10** | 30 | 100 |
|---|---|---|---|---|---|
| CV RMSE | 0.13027 | 0.12843 | **0.12766** | 0.12797 | 0.12980 |

- 呈 U 形：兩邊差，中間好。左邊（alpha 細）係 overfit，右邊（alpha 大）係 underfit。**呢個 U 形本身就係 bias-variance trade-off 嘅示範**（bias 同 variance 係咩 → Part 0 §4.4），可以畫入報告。
- 結果確認咗原本估嘅 10 已經係最好，所以 6b 同 6 分數一樣。

**誠實嘅限制：** 用同一個 CV 去揀 alpha 再報分數，分數會有少少樂觀，因為我哋揀咗「喺呢 5 個 fold 表現最好」嘅值。不過只試咗 5 個值，影響好細。

---

## 8. 最後一格：Part 3 結果表

```python
part3 = results[results.step.isin(["1", "2", "3", "4", "5", "6", "6b"])].copy()
part3["improvement"] = -part3.cv_rmse.diff()
part3
```
- `.isin([...])` 會揀出 step 喺清單入面嘅行。
- `.diff()` 計每行同上一行嘅差（→ Part 0 §3.3）；前面加負號，令「進步」變成正數，睇起嚟直觀啲。
- **點睇呢張表：**
  - `improvement` 係**絕對差**（例如 0.018823），唔係 §0.3 嗰個百分比。
  - 第一行係 NaN，因為冇上一行可以比較。
  - 6b 顯示 `-0.000000`，即係 0。
- 呢張表可以直接放入報告。

---

## 9. 交 Kaggle 嘅建議次序

| Submission | 檔案 | 對應 Step | CV |
|---|---|---|---|
| S1 | `s1_ridge_raw.csv` | 1 | 0.1553 |
| S2 | `s2_new_features.csv` | 4 | 0.1512 |
| S3 | `s3_log_features.csv` | 5 | 0.1324 |
| S4 | `s4_no_outliers.csv` | 6b | 0.1277 |

- 按次序交，每次記低 Kaggle 分數。
- **S2 → S3 應該係最大嘅進步**，啱啱好用嚟答「best improvement」嗰條必答題，因為一次只改咗一樣嘢（log 面積），可以清楚解釋原因。
- 如果 Kaggle 分數同 CV 嘅方向唔一致，都係一個值得寫嘅發現。

---

## 10. 講師可能會問

**Q：你哋最有用嘅 feature engineering 係咩？點解？**
A：對偏斜嘅面積欄取 log1p，CV 跌咗 12.4%。原因係樓價同面積係比例關係：log 價錢對 log 面積先係直線，配合預測 log 價錢，就符合線性 model 嘅假設。同時，log 亦削弱咗極端大面積對 model 嘅影響。

**Q：你哋加嘅 TotalSF 等新特徵冇乜效果，點解仲保留？**
A：對 Ridge 嚟講，佢哋係現有欄嘅線性組合，所以冇新資訊，結果亦符合呢個預期。但取 log 之後，log(總面積) 就唔再係線性組合；而樹 model 每次只睇一欄，砌唔出總和。所以 Part 4 嘅樹 model 會受惠。

**Q：點解唔一開始就將所有 NA 填 0 或者刪走？**
A：NA 有兩種意思（冇 / 唔知）。填 0 會將「冇車房」同「車房面積唔知」混淆；刪走就會損失有用嘅行或欄。我哋用另一欄去證實 NA 嘅意思，再分開處理。

**Q：Step 3 效果咁細，係咪白做？**
A：唔係。佢效果細，因為 pipeline 嘅 imputer 已經隱含地將 NA 當成一個類別。呢個 step 嘅價值在於令數據意思正確、可解釋，亦都證實咗「pipeline 已經處理得幾好」呢個 insight。

**Q：揀 skew > 0.75 嘅欄算唔算 leakage？**
A：只用咗特徵，冇用 SalePrice，所以冇洩漏答案。嚴格嚟講，用全部 train 計 skew 會睇到驗證屋嘅特徵分佈，但佢只係決定「邊欄取 log」，影響可以忽略。如果要完全嚴謹，可以將 skew 計算放入 pipeline。

**Q：拎走 outlier 點樣確保唔係作弊？**
A：有理由（Partial sale、大屋賤賣）；只喺訓練拎走；評分照計；有 CV 證明有效（−3.6%）。

**Q：alpha 點揀？**
A：喺對數尺度上試 1 至 100，用 5-fold CV 揀最細 RMSE 嘅值，結果係 10。分數曲線係 U 形，顯示咗 overfitting 同 underfitting 之間嘅平衡。

**Q：做呢個 project 遇到咩困難？**（個人問題）
A：可以講 Step 6 嘅 bug：取 log 之後，舊嘅 outlier 規則失效咗，但程式冇報錯，只係「靜靜雞」冇效果。透過留意到分數完全冇變而發現問題，學識咗結果異常時要追查原因，同埋要加自動檢查。

---

## 課堂連結

- Data preprocessing：data transformation（log）、missing values、outlier detection
- Feature engineering / feature construction
- Regression：regularization、hyperparameter tuning
- Model evaluation：cross-validation、bias-variance trade-off
