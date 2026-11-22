# Part 0：基礎知識（Part 1–5 都會用到）

> 呢份筆記唔假設你識任何嘢。Part 1–5 嘅筆記遇到新概念，會寫「→ Part 0 §X」，就係指返呢度嘅章節。
> 唔使一次讀晒，遇到唔識先返嚟查都得。

---

## §1 Python 基本

### §1.1 變數（variable）

```python
alpha = 10
name = "Ridge"
```

`=` 嘅意思係「將右邊嘅值放入左邊個名」，**唔係數學上嘅相等**。之後用 `alpha` 就等於用緊 10。

### §1.2 幾種常見嘅「容器」

| 寫法 | 名 | 例子 | 用途 |
|---|---|---|---|
| `[a, b, c]` | list（清單） | `[1, 3, 10, 30]` | 一串有次序嘅嘢 |
| `(a, b)` | tuple | `("GarageArea", [...], "NoGarage")` | 同 list 差唔多，但唔改得，通常用嚟「打包」幾樣相關嘅嘢 |
| `{key: value}` | dict（字典） | `{1: 0.130, 3: 0.128}` | 用一個 key 查一個值，例如 `alpha_scores[3]` 會攞到 0.128 |

**位置由 0 開始數**：`[10, 20, 30][0]` 係 10。`[-1]` 係最後一個。

### §1.3 Function（函數）

```python
def add_one(x):
    return x + 1

add_one(5)   # → 6
```

- `def` 嘅意思係「定義一個 function」。
- `x` 係 **參數**，即係俾 function 嘅輸入。
- `return` 嘅意思係「交返呢個結果出去」。
- 縮排（行頭嗰 4 個空格）好重要，Python 靠縮排嚟分邊啲行屬於呢個 function。

**點解要寫 function？** 同一套步驟要對 train 同 test 各做一次。寫一次、用兩次，就保證兩邊做法一模一樣。

**預設值：** `def cv_evaluate(model, X, y, X_test, target="log")`。冇特別講嘅話，`target` 就係 `"log"`；想改就寫 `target="raw"`。呢種寫法叫 **keyword argument**。

### §1.4 `for` 迴圈

```python
for alpha in [1, 3, 10]:
    print(alpha)
```

逐個攞出嚟，每次放入 `alpha`，然後執行縮排入面嘅嘢。上面會印出 1、3、10。

**一次拆幾個值（unpacking）：**

```python
for size_column, columns, label in ABSENT_RULES:
```

`ABSENT_RULES` 入面每一項都係一個有 3 樣嘢嘅 tuple，呢行每次會將嗰 3 樣分別放入 3 個名度。

### §1.5 `if` / `else`

```python
if target == "log":
    ...    # 條件成立先做
else:
    ...    # 唔成立就做呢度
```

- `==` 係「是否相等」嘅比較，結果係 True 或者 False。
- 一個 `=` 就係 §1.1 嘅「放入」，唔好撈亂。

### §1.6 Method（方法）同 `.`

`train.head()` 嘅意思係「叫 `train` 呢個物件做 `head` 呢個動作」。

method 可以一路串落去，由左至右逐個做：

```python
train.isna().sum()
```

1. `train.isna()`：每格變成 True / False；
2. `.sum()`：數每欄有幾多個 True。

### §1.7 其他常見寫法

| 寫法 | 意思 |
|---|---|
| `# ...` | 註解，Python 會忽略，係寫俾人睇嘅 |
| `import numpy as np` | 載入一個套件，之後用 `np` 呢個短名稱叫佢 |
| `from sklearn.linear_model import Ridge` | 由套件入面只攞 `Ridge` 一樣嘢 |
| `f"score = {x:.3f}"` | f-string：將變數 x 嘅值塞入字串，顯示 3 個小數位 |
| `_` | 「呢個值我唔要」嘅慣例名 |
| `assert 條件` | 條件唔成立就即刻報錯停低，用嚟做安全檢查 |
| `global results` | 喺 function 入面改外面嗰個 `results` |
| `x += 1` | 即係 `x = x + 1` |

---

## §2 NumPy（`np`）：一次過計一大堆數

### §2.1 Array 同向量化計算

```python
a = np.array([100, 200, 300])
a * 2          # → [200, 400, 600]，三個數一次過計，唔使寫 for
np.log(a)      # → 每個數各自取 log
```

### §2.2 常用 function

| Function | 做咩 | 例子 |
|---|---|---|
| `np.log(x)` | 自然對數 | `np.log(100000)` ≈ 11.5 |
| `np.exp(x)` | log 嘅反向 | `np.exp(11.5)` ≈ 100000 |
| `np.log1p(x)` | log(1 + x)，x 係 0 都計得 | `np.log1p(0)` = 0 |
| `np.sqrt(x)` | 開方 | |
| `np.mean(x)` | 平均 | |
| `np.clip(x, a, b)` | 細過 a 當 a，大過 b 當 b（`None` 即係冇上限） | |
| `np.where(條件, A, B)` | 條件成立就揀 A，唔成立就揀 B | |
| `np.zeros(n)` | n 個 0 | |

### §2.3 Boolean mask（用 True/False 揀行）

```python
price = np.array([100, 500, 200])
big = price > 150          # → [False, True, True]
price[big]                 # → [500, 200]
```

- 組合條件：`&`（同埋）、`|`（或者）、`~`（唔係）。
- 每個條件都要加括號：`(a > 1) & (b < 2)`。

---

## §3 pandas（`pd`）：處理表格

### §3.1 DataFrame 同 Series

- **DataFrame** 就係一張表，好似 Excel 咁，例如 `train`。
- **Series** 就係 DataFrame 入面嘅一欄，例如 `train["SalePrice"]`。

`train.SalePrice` 同 `train["SalePrice"]` 係同一樣嘢。不過如果欄名用數字開頭（例如 `"1stFlrSF"`），就只可以用方括號。

### §3.2 揀行揀欄

| 寫法 | 意思 |
|---|---|
| `df["A"]` | 攞 A 欄 |
| `df[["A", "B"]]` | 攞 A、B 兩欄（注意有兩層方括號） |
| `df.loc[條件, "A"]` | 揀「符合條件嘅行」嘅 A 欄，可以用嚟讀，亦可以用嚟寫 |
| `df.iloc[[0, 5, 9]]` | 用**位置**揀第 0、5、9 行 |
| `df.drop(columns="Id")` | 拎走 Id 欄，回傳一張新表，原本嗰張唔變 |

`.loc` 用嚟寫入：

```python
df.loc[absent & df[column].isna(), column] = label
```

意思係「喺 absent 而且 column 係缺失嘅嗰啲行，將 column 改做 label」。

### §3.3 常用 method

| Method | 做咩 |
|---|---|
| `.shape` | (行數, 欄數) |
| `.head()` | 頭 5 行 |
| `.isna()` | 每格係咪缺失（True/False） |
| `.fillna(v)` | 將缺失填做 v |
| `.astype(str)` | 轉做文字 |
| `.sum()` / `.sum(axis=1)` | 每欄加總 / 每行加總 |
| `.skew()` | 偏度（§5.3） |
| `.corr()` | 相關系數 |
| `.diff()` | 同上一行嘅差 |
| `.copy()` | 整一個獨立副本 |
| `.clip(lower=0)` | 細過 0 嘅當 0 |

### §3.4 點解 function 第一行要寫 `df = df.copy()`

pandas 嘅 DataFrame 傳入 function 之後，如果直接改佢，**外面原本嗰張表都會一齊被改**。

`.copy()` 會先整一個副本，之後只改副本。咁樣 `X` 同 `X3` 先可以同時存在、互不影響，你亦可以重跑任何一個 cell 都唔會撈亂。

---

## §4 Machine learning 基本概念

### §4.1 X 同 y

- **X（features，特徵）**：用嚟估嘅資料，例如面積、年份、地區。每行一間屋，每欄一個特徵。
- **y（target，目標）**：要估嘅答案，即係 SalePrice。
- **Regression（回歸）**：y 係一個數字。如果 y 係類別（例如「貴 / 平」），就叫 classification。

### §4.2 Model、`fit`、`predict`

model 就係「一條由 X 計出 y 嘅規則」，入面有啲數字要從數據學返嚟。

```python
model.fit(X_train, y_train)    # 學：睇住答案調整入面嘅數字
model.predict(X_new)           # 用：俾新嘅 X，計出估計嘅 y
```

sklearn 入面所有 model 都係用呢兩個 method，所以換 model 唔使改其他 code。

### §4.3 Linear regression 同 coefficient

$$\hat y = b_0 + b_1 x_1 + b_2 x_2 + \dots$$

- 每個 $b$ 叫 **coefficient（係數）**，即係每個特徵嘅「權重」，`fit` 就係搵出最好嘅 $b$。
- 例子：$\hat y = 50000 + 100 \times \text{面積}$，代表每多 1 平方呎，估價就多 $100。
- **限制：** 線性 model 只可以將特徵「乘一個數再加埋」。佢唔識自己諗出「面積越大，每呎價錢越低」呢種彎嘅關係，所以要靠我哋喺 feature engineering 度幫佢。

### §4.4 Overfitting 同 underfitting

- **Overfitting（過度擬合）**：model 死記咗訓練數據，連啲隨機雜訊都記埋，所以訓練數據估得好準，新數據就估得差。好似背晒舊試卷答案，但一換題目就唔識。
- **Underfitting（擬合不足）**：model 太簡單，連訓練數據都估唔準。
- CV（Part 2）就係用嚟量度「對新數據有幾準」，所以 CV 可以捉到 overfitting。

### §4.5 Parameter 同 hyperparameter

| | 由邊個決定 | 例子 |
|---|---|---|
| **Parameter** | model 喺 `fit` 時自己學 | Ridge 嘅 coefficient |
| **Hyperparameter** | 我哋喺 `fit` 之前設定 | Ridge 嘅 `alpha` |

揀 hyperparameter 嘅方法就係：試幾個值，用 CV 比較，揀最好嗰個。呢個過程叫 **tuning（調參）**。

### §4.6 sklearn 嘅 Transformer 同 Pipeline

- **Transformer**：負責改數據，唔負責估價。例如 `SimpleImputer`（填缺失）、`StandardScaler`（標準化）、`OneHotEncoder`。
  - 佢哋用 `fit` 學嘢（例如學 median 係幾多），再用 `transform` 套用。
- **Pipeline**：將幾個 transformer 同最後嘅 model 串成一條線。
  - `pipeline.fit(X, y)`：逐個 transformer 喺**同一份訓練數據**上 fit 同 transform，最後 fit model。
  - `pipeline.predict(X_new)`：逐個 transform（用 fit 時學返嚟嘅數字），最後 predict。
- 好處：整條 pipeline 可以當成一個 model 用，自動避開 leakage（Part 2 §1.3）。

### §4.7 RMSE

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum (\text{真實} - \text{估計})^2}$$

計法：計每個誤差 → 平方（令負數變正，大錯罰得更重）→ 平均 → 開方（變返原本單位）。**越細越好。**

---

## §5 Log 嘅數學直覺

### §5.1 Log 係咩

$\log(x)$ 回答一個問題：**「e（約 2.718）要乘自己幾多次先等於 x？」**

- 你唔使記數值，只要記住佢嘅**性質**。
- `np.log` 係自然對數（底數 e）。

### §5.2 最重要嘅性質：乘數變加數

$$\log(a \times b) = \log a + \log b$$

| 原本 | 取 log 之後 |
|---|---|
| 價錢 ×1.1（貴 10%） | log 價錢 +0.095 |
| 100,000 → 110,000 | 11.513 → 11.608（+0.095） |
| 1,000,000 → 1,100,000 | 13.816 → 13.911（**一樣係 +0.095**） |

所以喺 log 尺度上，「貴 10%」無論係平樓定貴樓都係同一個距離。呢個就係 Kaggle 用 log 計分、同埋我哋預測 log 價錢嘅原因。

### §5.3 Skewness（偏度）同 log 點樣「拉直」數據

- 右偏：大部份數值集中喺細嘅一邊，少數極大，例如面積、樓價。
- log 會「壓扁大數、拉開細數」：
  - 100 → 4.6
  - 1,000 → 6.9
  - 10,000 → 9.2

  原本差 10 倍、100 倍嘅距離，變成每次只係 +2.3。結果長尾巴縮短咗，分佈就接近對稱。
- 慣例：skew > 0.75 就當係「明顯偏」，要考慮取 log。0.75 唔係定律，只係常用門檻。

### §5.4 `log1p`

`log1p(x) = log(1 + x)`。

- 面積可以係 0（例如冇 porch），但 `log(0)` 係負無限大，會爆。
- 用 log1p 嘅話，0 → 0，而大數嘅結果同 log 幾乎一樣。
- 反向運算係 `expm1`。不過我哋只係對**特徵**取 log1p，唔使轉返轉頭，所以用唔到 `expm1`。
