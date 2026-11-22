# Part 0：基礎知識（Part 1–5 都會用到）

> 呢份筆記唔假設你識任何嘢。Part 1–5 嘅筆記遇到新概念，會寫「→ Part 0 §X」，就係指返呢度嘅章節。
> 唔使一次讀晒，遇到唔識先返嚟查都得。
>
> **編號說明：** 筆記入面用「§」嘅編號（例如「Part 1 §1.5」）係指**筆記**嘅章節。notebook 入面嘅英文標題（例如「1.4 Outliers」）係另一套編號。兩套唔同，筆記提到 notebook 嘅格時會寫「Cell 5」。

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
| `{a, b, c}` | set（集合） | `{"2", "3", "4"}` | 冇次序、冇重複，通常用 `x in 集合` 查「係咪喺入面」。留意：有 `:` 嘅 `{}` 係 dict，冇 `:` 嘅係 set |

**位置由 0 開始數**：`[10, 20, 30][0]` 係 10。`[-1]` 係最後一個，`[-2]` 係倒數第二個。

**Slicing（切片）：** 用 `[開始:結束]` 攞一段，**唔包結束嗰個位置**。

| 寫法 | 例子 `a = [10, 20, 30, 40]` | 結果 |
|---|---|---|
| `a[1:3]` | 第 1 至第 2 個 | `[20, 30]` |
| `a[:2]` | 由頭到第 1 個 | `[10, 20]` |
| `a[:-1]` | 除咗最後一個 | `[10, 20, 30]` |
| `a[::-1]` | 倒轉 | `[40, 30, 20, 10]` |

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

**預設值：** `def cv_evaluate(model, X, y, X_test, target="log", drop_outliers=False)`。冇特別講嘅話，`target` 就係 `"log"`，`drop_outliers` 就係 `False`；想改就寫 `target="raw"` 或者 `drop_outliers=True`。呢種寫法叫 **keyword argument**。

**Docstring（說明文字）：** function 第一行如果係用三個引號包住嘅字（`"""…"""`），就係俾人睇嘅說明，Python 唔會執行。

**Lambda（一行 function）：** `lambda a: Lasso(alpha=a)` 等於：

```python
def make(a):
    return Lasso(alpha=a)
```

只係冇名、寫成一行。當你要將「一個動作」交俾另一個 function 用時就好方便（Part 4 嘅 `tune_alpha` 會用到）。

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

`ABSENT_RULES` 入面每一項都係一個有 3 樣嘢嘅 tuple，呢行每次會將嗰 3 樣分別放入 3 個變數。

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
| `print(x)` | 喺畫面印出 x |
| `len(x)` | x 有幾多個（list 有幾多項、DataFrame 有幾多行） |
| `range(n)` | 0, 1, …, n−1，常用喺 `for i in range(n)` |
| `for i, v in enumerate(x)` | 逐個攞出嚟，同時俾埋位置 i（0, 1, 2…） |
| `min(d, key=d.get)` | 喺 dict d 入面搵「值最細」嗰個 key（Part 3 §7.2 有詳細例子） |

**f-string 嘅格式：**

| 寫法 | 意思 | 例子 |
|---|---|---|
| `{x:.5f}` | 5 個小數位 | 0.12766 |
| `{x:+.5f}` | 5 個小數位，**一定顯示正負號** | +0.00014、−0.01882 |
| `{x:.0f}` | 唔要小數 | 83 |
| `{x:>3}` | 靠右、佔 3 格（令幾行對齊） | `  1`、` 10`、`100` |

### §1.8 縮短咗嘅 for：comprehension 同一行 if

呢幾種寫法本質上都係 for 迴圈或者 if，只係寫成一行。

**List comprehension：** 逐個計一樣嘢，結果組成 list。

```python
[x * 2 for x in [1, 2, 3]]          # → [2, 4, 6]
```

等於：

```python
result = []
for x in [1, 2, 3]:
    result.append(x * 2)
```

**Dict comprehension：** 結果係 dict。

```python
{name: len(name) for name in ["xgb", "svr"]}    # → {"xgb": 3, "svr": 3}
```

**加條件：** `[x for x in 清單 if 條件]`，只保留條件成立嘅。

**`next(x for x in 清單 if 條件)`：** 攞**第一個**符合條件嘅。括號入面嗰樣叫 generator expression，佢唔會一次過計晒，而係一個一個咁試，搵到就停。如果全部唔符合，`next` 會報錯（或者回傳你俾佢嘅預設值，例如 `next(…, None)` 就回傳 None）。

**一行 if：** `A if 條件 else B`。條件成立就係 A，否則係 B。

```python
"white" if value > 0.93 else "black"
```

可以串落去：`A if 條件1 else B if 條件2 else C`，由左至右逐個條件試。

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

**Index（行名）：** 每張 DataFrame / Series 左邊都有一欄「行名」，叫 index。

- `train` 嘅 index 係 0, 1, 2, …（行號）。
- 但 `train.isna().sum()` 嘅結果係一條 Series，佢嘅 index 係 **train 嘅欄名**（"PoolQC"、"Alley"…），值係每欄缺幾多格。
- `.index` 攞出行名；`.values` 攞出入面嘅數值，變成 numpy array（冇咗行名，純粹一串數）。
- `set_index("step")`：將某一欄變成行名，之後可以用 `表["5"]` 咁查。
- `reset_index(drop=True)`：將行名重新編成 0, 1, 2…。

**NaN：** 即係 Not a Number，係 pandas / numpy 用嚟表示「缺失」嘅特別值，`.isna()` 就係搵佢。NaN 同任何數相加都係 NaN；`NaN == 0` 係 False。

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
| `.corr()` | 相關係數 |
| `.diff()` | 同上一行嘅差 |
| `.copy()` | 整一個獨立副本 |
| `.clip(lower=0)` | 細過 0 嘅當 0 |
| `.eq(v)` | 即係 `== v`，每格係咪等於 v |
| `.isin([...])` | 每格係咪喺清單入面 |
| `.sort_values("A", ascending=False)` | 按 A 欄排序；`ascending=False` 即係由大到細 |
| `.tail(n)` | 最後 n 行 |
| `.round(n)` | 取 n 個小數位 |
| `.tolist()` | 變成普通 list |
| `.value_counts()` | 數每個值出現幾多次 |
| `.select_dtypes("number")` | 只揀數字欄（`exclude=` 就係排除） |
| `.to_csv(路徑, index=False)` | 存成 CSV；`index=False` 即係唔好將行名都寫入去（否則 Kaggle 會見到多咗一欄） |
| `pd.concat([表1, 表2])` | 將兩張表上下疊埋 |
| `.empty` | 張表係咪完全冇行 |
| `.all()` | 係咪全部都係 True |
| `np.isfinite(x)` | 每個數係咪正常數字（唔係 NaN、唔係無限大） |

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

```
估計價錢 = b0 + b1 × 特徵1 + b2 × 特徵2 + …
```

- 每個 `b` 叫 **coefficient（係數）**，即係每個特徵嘅「權重」，`fit` 就係搵出最好嘅 `b`。
- 例子：`估計價錢 = 50000 + 100 × 面積`，代表每多 1 平方呎，估價就多 100 蚊。
- **限制：** 線性 model 只可以將特徵「乘一個數再加埋」。佢唔識自己諗出「面積越大，每呎價錢越低」呢種彎嘅關係，所以要靠我哋喺 feature engineering 度幫佢。

### §4.4 Overfitting 同 underfitting

- **Overfitting（過度擬合）**：model 死記咗訓練數據，連啲隨機雜訊都記埋，所以訓練數據估得好準，新數據就估得差。好似背晒舊試卷答案，但一換題目就唔識。
- **Underfitting（擬合不足）**：model 太簡單，連訓練數據都估唔準。
- CV（Part 2）就係用嚟量度「對新數據有幾準」，所以 CV 可以捉到 overfitting。

**Bias 同 variance（兩種錯誤嘅來源）：**

| | 意思 | 例子 | 對應 |
|---|---|---|---|
| **Bias（偏差）** | model 太簡單，次次都錯同一個方向 | 用直線去 fit 一條彎線 | underfitting |
| **Variance（變異）** | model 太跟訓練數據，換一份數據結果就大變 | 一棵好深嘅決策樹 | overfitting |

**Bias-variance trade-off：** model 越複雜，bias 越細，但 variance 越大，所以要搵中間最平衡嘅位置。Part 3 調 alpha 嘅 U 形曲線就係例子：alpha 太細 → variance 大；alpha 太大 → bias 大。

### §4.5 Parameter 同 hyperparameter

| | 由邊個決定 | 例子 |
|---|---|---|
| **Parameter** | model 喺 `fit` 時自己學 | Ridge 嘅 coefficient |
| **Hyperparameter** | 我哋喺 `fit` 之前設定 | Ridge 嘅 `alpha` |

揀 hyperparameter 嘅方法就係：試幾個值，用 CV 比較，揀最好嗰個。呢個過程叫 **tuning（調參）**。

**Regularization（正則化）：** 喺「誤差」之外再加一項 **penalty（懲罰）**，逼 model 唔好太複雜，即係減少 variance、防止 overfitting。Ridge 嘅 `alpha` 就係正則化嘅強度：alpha 越大，懲罰越重，model 越簡單。筆記入面「penalty」、「懲罰」、「正則化」講嘅都係呢樣嘢。

### §4.6 sklearn 嘅 Transformer 同 Pipeline

- **Transformer**：負責改數據，唔負責估價。例如 `SimpleImputer`（填缺失）、`StandardScaler`（標準化）、`OneHotEncoder`。
  - 佢哋用 `fit` 學嘢（例如學 median 係幾多），再用 `transform` 套用。

**用到嘅三個統計概念：**

| 概念 | 意思 | 例子 |
|---|---|---|
| **Median（中位數）** | 由細到大排好，正中間嗰個 | 1, 2, **3**, 100, 200 嘅 median 係 3（平均係 61，被 200 拉高咗） |
| **Standard deviation（標準差）** | 數據平均離平均值有幾遠，即係「散得幾開」 | 面積平均 1500 呎、標準差 500 呎，代表大部份屋喺 1000–2000 呎之間 |
| **Standardization（標準化）** | `(x − 平均) ÷ 標準差` | 上面例子：2000 呎 → (2000−1500)/500 = **1**；1000 呎 → **−1** |

標準化之後，每個特徵平均都係 0、標準差都係 1，即係全部喺同一把尺上。
- **Pipeline**：將幾個 transformer 同最後嘅 model 串成一條線。
  - `pipeline.fit(X, y)`：逐個 transformer 喺**同一份訓練數據**上 fit 同 transform，最後 fit model。
  - `pipeline.predict(X_new)`：逐個 transform（用 fit 時學返嚟嘅數字），最後 predict。
- 好處：整條 pipeline 可以當成一個 model 用，自動避開 leakage（Part 2 §1.3）。

### §4.7 RMSE

```
RMSE = √( 平均( (真實 − 估計)² ) )
```

計法：計每個誤差 → 平方（令負數變正，大錯罰得更重）→ 平均 → 開方（變返原本單位）。**越細越好。**

---

## §5 Log 嘅數學直覺

### §5.1 Log 係咩

`log(x)` 回答一個問題：**「e（約 2.718）要乘自己幾多次先等於 x？」**

- 你唔使記數值，只要記住佢嘅**性質**。
- `np.log` 係自然對數（底數 e）。

### §5.2 最重要嘅性質：乘數變加數

```
log(a × b) = log(a) + log(b)
```

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

---

## §6 Matplotlib（`plt`）：畫圖

| 寫法 | 意思 |
|---|---|
| `fig, ax = plt.subplots(figsize=(7, 5))` | 開一幅畫布（`fig`），入面有一格圖（`ax`），闊 7 吋高 5 吋 |
| `fig, axes = plt.subplots(1, 2)` | 1 行 2 格，`axes[0]`、`axes[1]` 係兩格圖 |
| `ax.hist(...)` / `ax.bar(...)` / `ax.barh(...)` / `ax.scatter(...)` | 直方圖 / 直棒圖 / 橫棒圖 / 散點圖 |
| `ax.set(title=..., xlabel=..., ylabel=..., xlim=..., ylim=...)` | 一次過設標題、軸名、軸範圍 |
| `ax.text(x, y, "字", ha="center", va="center")` | 喺 (x, y) 寫字；`ha` / `va` 係水平 / 垂直對齊方式 |
| `ax.legend()` | 顯示圖例（邊種顏色或記號代表咩） |
| `fig.tight_layout()` | 自動調整位置，費事啲字重疊 |
| `fig.savefig(路徑, dpi=120)` | 存成圖片檔（dpi 越大越清） |
| `plt.show()` | 喺 notebook 顯示 |

**Figure 同 Axes 嘅分別：** `fig` 係成張紙，`ax` 係紙上面其中一格圖。一張紙可以有幾格（`axes`）。畫嘢係對 `ax` 做，存檔係對 `fig` 做。

