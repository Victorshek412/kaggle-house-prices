# Part 0：基礎知識（Part 1–5 都會用到）

> **點用呢份筆記：** 每一個概念都會跟住咁嘅次序講：
> 1. 一句講佢係咩；
> 2. 一段好短嘅 code（大部份用真實樓價數據）；
> 3. **「跑出嚟會見到」**：呢段 code 真正跑出嚟嘅結果；
> 4. 逐樣解釋個結果點睇。
>
> 你可以喺 Kaggle notebook 開一格新 cell，自己貼段 code 落去跑，對下個結果係咪一樣。唔使一次讀晒，Part 1–5 遇到唔識嘅嘢，佢哋會寫「→ Part 0 §X」，到時返嚟查就得。
>
> **編號說明：** 筆記入面用「§」嘅編號（例如「Part 1 §1.5」）係指**筆記**嘅章節。notebook 入面嘅英文小標題（例如「1.4 Outliers」）係另一套編號。筆記提到 notebook 嘅格時會寫「Cell 5」。

---

## §1 Python 基本

### §1.1 變數（variable）

**一句講晒：** 變數就係一個「名」，用嚟裝住一個值，等你之後可以用個名叫返個值出嚟。

```python
price = 180000
name = "Ridge"
price_in_thousands = price / 1000
print(price)
print(name)
print(price_in_thousands)
```

跑出嚟會見到：

```text
180000
Ridge
180.0
```

**點睇：**
- `price = 180000`：將 180000 呢個數放入一個叫 `price` 嘅名度。`=` 唔係數學嘅「相等」，而係「**放入**」。
- `"Ridge"`：有引號嘅係**文字**（叫 string）；冇引號嘅 180000 係數字。
- `price / 1000`：用個名做計算，等於 180000 ÷ 1000。
- `print(...)`：將括號入面嘅嘢顯示喺畫面。

### §1.2 幾種常見嘅「容器」：list、tuple、dict、set

成日要裝**一堆**值，Python 有 4 種常用容器：

```python
alphas = [1, 3, 10, 30]                       # list：有次序嘅一串
rule = ("GarageArea", "NoGarage")             # tuple：打包幾樣相關嘅嘢
alpha_scores = {1: 0.130, 3: 0.128, 10: 0.1277}   # dict：用 key 查 value
feature_steps = {"2", "3", "4"}               # set：冇次序、冇重複
print(alphas[0], alphas[-1])
print(rule[1])
print(alpha_scores[10])
print("3" in feature_steps, "7" in feature_steps)
```

跑出嚟會見到：

```text
1 30
NoGarage
0.1277
True False
```

**點睇：**
- `alphas[0]` 攞**第一個**（位置由 0 開始數！），所以係 1；`alphas[-1]` 攞**最後一個**，係 30。
- `rule[1]` 係 tuple 入面第二樣嘢，即係 "NoGarage"。tuple 同 list 差唔多，分別係 tuple 建立咗之後唔可以改。
- `alpha_scores[10]`：dict 唔係用位置，而係用 **key** 去查。key 10 對應嘅 value 係 0.1277。好似查字典：查「10」呢個字，就見到佢嘅解釋。
- `"3" in feature_steps`：問「"3" 喺唔喺個 set 入面？」，係就 True，唔係就 False。
- 分辨方法：`[ ]` 係 list，`( )` 係 tuple，`{a: b}`（有冒號）係 dict，`{a, b}`（冇冒號）係 set。

**Slicing（切片）：攞一段出嚟**

```python
a = [10, 20, 30, 40]
print(a[1:3])    # 第 1 至第 2 個（唔包第 3 個）
print(a[:2])     # 由頭到第 1 個
print(a[:-1])    # 除咗最後一個
print(a[::-1])   # 倒轉
```

跑出嚟會見到：

```text
[20, 30]
[10, 20]
[10, 20, 30]
[40, 30, 20, 10]
```

**點睇：** `[開始:結束]` 包「開始」但**唔包「結束」**，所以 `a[1:3]` 係位置 1 同 2，即係 20、30。`[::-1]` 嘅 -1 代表「一步步倒行」，所以成個倒轉。之後畫橫棒圖時會成日用 `[::-1]`，令最大嘅排最上面。

### §1.3 Function（函數）

**一句講晒：** function 就係將一段做法包起嚟、改個名，之後一句就可以叫佢做成段嘢。

```python
def add_fee(price, fee=5000):
    """Return the price plus an agent fee."""
    return price + fee

print(add_fee(180000))
print(add_fee(180000, fee=8000))
```

跑出嚟會見到：

```text
185000
188000
```

**逐行點睇：**
- `def add_fee(price, fee=5000):`
  - `def` 代表「定義一個 function」，個名叫 `add_fee`。
  - 括號入面係**參數**，即係 function 嘅輸入。
  - `fee=5000` 係**預設值**：冇講就用 5000。
- `"""..."""`：三個引號包住嘅叫 **docstring**，只係俾人睇嘅說明，Python 唔會執行。
- `return ...`：將計好嘅結果「交返出去」。
- **縮排**（行頭 4 個空格）好重要：Python 靠佢分邊幾行屬於呢個 function。
- `add_fee(180000)`：冇講 fee，用預設 5000，得 185000。
- `add_fee(180000, fee=8000)`：講明 fee=8000，得 188000。呢種「名=值」嘅寫法叫 **keyword argument**。

**喺我哋嘅 project：** `cv_evaluate(model, X, y, X_test, target="log", drop_outliers=False)` 都係咁。平時唔講就係 `target="log"`、`drop_outliers=False`；Step 0、1 寫 `target="raw"`，Step 6 開始寫 `drop_outliers=True`。

**點解要寫 function？** 同一套步驟要對 train 同 test 各做一次。寫一次、用兩次，就保證兩邊做法一模一樣。

**Lambda：一行、冇名嘅 function**

```python
make_double = lambda a: a * 2
print(make_double(5))
```

跑出嚟會見到：

```text
10
```

`lambda a: a * 2` 等於：

```python
def make_double(a):
    return a * 2
```

只係寫成一行。當你要「將一個做法交俾另一個 function 用」時就好方便。Part 4 嘅 `tune_alpha(lambda a: Lasso(alpha=a), ...)` 就係將「俾我一個 alpha，我整一個 Lasso」呢個做法交俾 `tune_alpha`。

### §1.4 `for` 迴圈：逐個做一次

```python
for alpha in [1, 3, 10]:
    print("trying alpha =", alpha)
```

跑出嚟會見到：

```text
trying alpha = 1
trying alpha = 3
trying alpha = 10
```

**點睇：** 由 list 逐個攞出嚟，每次放入 `alpha`，然後做一次縮排入面嘅嘢。list 有 3 個，所以做 3 次。

**一次拆幾個值（unpacking）**

```python
rules = [("GarageArea", "NoGarage"), ("PoolArea", "NoPool")]
for size_column, label in rules:
    print(size_column, "->", label)
```

跑出嚟會見到：

```text
GarageArea -> NoGarage
PoolArea -> NoPool
```

**點睇：** `rules` 每一項都係一個有 2 樣嘢嘅 tuple。`for size_column, label in rules` 每次會自動將嗰 2 樣分別放入 `size_column` 同 `label` 兩個變數。Part 3 嘅 `ABSENT_RULES` 每項有 3 樣嘢，原理一樣。

**`enumerate`：一路行一路俾埋位置**

```python
for i, score in enumerate([0.155, 0.152, 0.151]):
    print(i, score)
```

跑出嚟會見到：

```text
0 0.155
1 0.152
2 0.151
```

**點睇：** `i` 係位置（0, 1, 2），`score` 係值。Part 5 畫圖時用佢喺每條棒頂寫分數。

**兩層 for**

```python
for C in [1, 10]:
    for gamma in [0.001, 0.003]:
        print("C =", C, "gamma =", gamma)
```

跑出嚟會見到：

```text
C = 1 gamma = 0.001
C = 1 gamma = 0.003
C = 10 gamma = 0.001
C = 10 gamma = 0.003
```

**點睇：** 外層每行一次，內層就行晒一輪，所以總共 2 × 2 = 4 次。Part 4 嘅 SVR grid search 就係咁試晒所有組合。

### §1.5 `if` / `else`：揀做邊樣

```python
target = "log"
if target == "log":
    print("learn log prices")
else:
    print("learn raw prices")
```

跑出嚟會見到：

```text
learn log prices
```

**點睇：**
- `==`（兩個等號）係**問**「係咪相等？」，結果係 True 或 False。
- 一個 `=` 係 §1.1 嘅「放入」，唔好撈亂。
- 條件成立就做 `if` 下面嗰段，否則做 `else` 下面嗰段。

### §1.6 Method（方法）同 `.`

**一句講晒：** `東西.動作()`，即係叫呢樣東西做某個動作。

```python
word = "nogarage"
print(word.upper())
print(word.upper().replace("NO", "NO_"))
```

跑出嚟會見到：

```text
NOGARAGE
NO_GARAGE
```

**點睇：** `word.upper()` 叫個字串變大楷。method 可以一路串落去，**由左至右**逐個做：先變大楷，再將 "NO" 換成 "NO_"。之後 `train.isna().sum()` 都係咁讀：先 `isna()`，再 `sum()`。

### §1.7 其他常見寫法

| 寫法 | 意思 |
|---|---|
| `# ...` | 註解，Python 會忽略，係寫俾人睇嘅 |
| `import numpy as np` | 載入 numpy 呢個套件，之後用 `np` 呢個短名叫佢 |
| `from sklearn.linear_model import Ridge` | 由套件入面只攞 `Ridge` 一樣嘢 |
| `_` | 「呢個值我唔要」嘅慣例名 |
| `assert 條件` | 條件唔成立就即刻報錯停低，用嚟做安全檢查 |
| `global results` | 喺 function 入面改**外面**嗰個 `results` |
| `x += 1` | 即係 `x = x + 1` |
| `len(x)` | x 有幾多個 |
| `range(n)` | 0, 1, …, n−1 |
| `None` | Python 嘅「冇嘢」 |

**f-string：將數字塞入文字，仲可以控制格式**

```python
score = 0.1276559
change = -0.0047304
seconds = 82.7
print(f"CV RMSE {score:.5f}")
print(f"change {change:+.5f}")
print(f"took {seconds:.0f} s")
print(f"alpha={3:>3}|")
print(f"alpha={100:>3}|")
```

跑出嚟會見到：

```text
CV RMSE 0.12766
change -0.00473
took 83 s
alpha=  3|
alpha=100|
```

**點睇：**
- 字串前面加 `f`，`{ }` 入面嘅嘢就會被換成真正嘅值。
- `:.5f`：5 個小數位。
- `:+.5f`：5 個小數位，**一定顯示正負號**。`record` 用呢個，所以你會見到 `+0.00014` 或者 `-0.01882`。
- `:.0f`：唔要小數（82.7 → 83，會四捨五入）。
- `:>3`：靠右、佔 3 格，令幾行嘅數字對齊（留意 3 前面多咗兩個空格）。

### §1.8 縮短咗嘅 for：comprehension 同一行 if

呢幾種寫法本質上都係 for 迴圈或者 if，只係寫成一行。睇到唔好驚，拆返做普通 for 就明。

**List comprehension**

```python
prices = [100000, 200000, 400000]
doubled = [p * 2 for p in prices]
cheap = [p for p in prices if p < 300000]
print(doubled)
print(cheap)
```

跑出嚟會見到：

```text
[200000, 400000, 800000]
[100000, 200000]
```

**點睇：** `[p * 2 for p in prices]` 讀法係「對 prices 入面每個 p，計 p × 2，全部放入一個新 list」。加 `if` 就只保留條件成立嘅。佢等於：

```python
doubled = []
for p in prices:
    doubled.append(p * 2)
```

**Dict comprehension**

```python
names = ["xgb", "svr"]
{name: len(name) for name in names}
```

跑出嚟會見到：

```text
{'xgb': 3, 'svr': 3}
```

**點睇：** 同上面一樣，只係結果係 dict：`key: value`。Part 5 嘅 `{name: y_log - oof[name] for name in names}` 就係「每個 model 名 → 佢嘅錯誤」。

**`next(...)`：攞第一個符合條件嘅**

```python
folders = ["/kaggle/input/abc", "../data", "data"]
exists = {"../data"}                          # pretend only this one exists
print(next((f for f in folders if f in exists), None))
print(next((f for f in folders if f == "nowhere"), None))
```

跑出嚟會見到：

```text
../data
None
```

**點睇：** 括號入面嗰段叫 generator expression，佢會逐個試，搵到第一個符合條件嘅就停，所以第一行印 `../data`。`next(…, None)` 嘅 `None` 係「一個都搵唔到」時嘅預設值，所以第二行印 `None`。Part 1 用呢個方法搵 data 放咗喺邊。

**一行 if：`A if 條件 else B`**

```python
value = 0.99
color = "white" if value > 0.93 else "black"
print(color)
```

跑出嚟會見到：

```text
white
```

**點睇：** 條件成立就係 A（"white"），否則係 B。Part 5 熱力圖就用呢個決定字嘅顏色。

---

## §2 NumPy（`np`）：一次過計一大堆數

### §2.1 Array 同向量化計算

```python
import numpy as np
prices = np.array([100000, 200000, 400000])
print(prices * 2)
print(np.log(prices))
```

跑出嚟會見到：

```text
[200000 400000 800000]
[11.51292546 12.20607265 12.89921983]
```

**點睇：** numpy array 好似 list，但可以**一次過計晒每一個**，唔使寫 for：`prices * 2` 每個都乘 2。`np.log(prices)` 每個都取 log。呢個叫**向量化（vectorization）**，又快又短。

### §2.2 常用 function

```python
x = np.array([0, 100, 10000])
print(np.log1p(x))
print(np.exp(np.log(180000)).round(2))
print(np.clip(np.array([-5, 50, 500]), 10, None))
print(np.where(np.array([3, 8, 1]) > 2, "big", "small"))
print(np.mean([100, 200, 300]), np.sqrt(16))
```

跑出嚟會見到：

```text
[0.         4.61512052 9.21044037]
180000.0
[ 10  50 500]
['big' 'big' 'small']
200.0 4.0
```

**逐行點睇：**
- `np.log1p(x)`：即係 log(1 + x)。0 → 0（如果用 `np.log(0)` 就會變負無限大，爆）。詳細 → §5.4。
- `np.exp(np.log(180000))`：exp 係 log 嘅反向，所以行一轉又變返 180000。我哋就係用 `np.exp` 將預測嘅 log 價錢變返做價錢。
- `np.clip(…, 10, None)`：細過 10 嘅當 10，`None` 即係冇上限。所以 -5 → 10。
- `np.where(條件, A, B)`：條件成立揀 A，否則揀 B，逐個做。
- `np.mean` 平均、`np.sqrt` 開方。
- 輸出入面數字後面嘅 `.`（例如 `4.`）只係代表「呢個係小數類型」，`4.` 即係 4.0。
- 小貼士：電腦計小數會有極細嘅誤差，例如 0.1 + 0.2 會得 0.30000000000000004。見到呢類長尾數唔使驚，用 `.round(2)` 就會變靚。

### §2.3 Boolean mask：用 True/False 揀出你要嘅

```python
price = np.array([100, 500, 200])
area = np.array([900, 5000, 1200])
big = area > 4000
print(big)
print(price[big])
print(price[~big])
print((area > 1000) & (price < 300))
```

跑出嚟會見到：

```text
[False  True False]
[500]
[100 200]
[False False  True]
```

**點睇：**
- `area > 4000` 逐個比較，得出一串 True/False，叫 **boolean mask**（好似一塊有窿嘅膠片，True 嘅位置先透光）。
- `price[big]`：用塊膠片揀，只要 True 嘅位置，所以得 500。
- `~big`：`~` 係「反轉」，True 變 False，所以揀返其他兩個。
- `&` 係「同埋」（兩個都要成立），`|` 係「或者」。每個條件**一定要加括號**。Part 1 搵 outlier 就係用 `(面積 > 4000) & (價錢 < 300000)`。

---

## §3 pandas（`pd`）：處理表格

下面用真實嘅 train 數據做例子。先讀入嚟（同 Part 1 Cell 1 一樣嘅讀法）：

```python
import pandas as pd
train = pd.read_csv("../house-prices-advanced-regression-techniques/train.csv",
                    keep_default_na=False, na_values=["NA", ""])
small = train[["Id", "GrLivArea", "GarageType", "SalePrice"]].head(5)
small
```

跑出嚟會見到：

```text
   Id  GrLivArea GarageType  SalePrice
0   1       1710     Attchd     208500
1   2       1262     Attchd     181500
2   3       1786     Attchd     223500
3   4       1717     Detchd     140000
4   5       2198     Attchd     250000
```

### §3.1 DataFrame、Series、index

**點睇上面個結果：**
- 成張表叫 **DataFrame**，好似 Excel。
- 最左邊嗰欄 0、1、2、3、4 唔係數據，而係**行名（index）**。
- 上面嘅 `Id`、`GrLivArea`……係**欄名**。

**Series：DataFrame 入面嘅一欄**

```python
small["SalePrice"]
```

跑出嚟會見到：

```text
0    208500
1    181500
2    223500
3    140000
4    250000
Name: SalePrice, dtype: int64
```

**點睇：** 攞一欄出嚟，就係一條 **Series**：左邊仍然係 index（0–4），右邊係數值，最底仲話你知欄名同資料類型（int64 = 整數）。`small.SalePrice` 同 `small["SalePrice"]` 一樣；不過欄名以數字開頭（例如 `"1stFlrSF"`）就只可以用方括號。

**Index 唔一定係 0, 1, 2：**

```python
missing = train[["PoolQC", "Alley", "LotFrontage", "Street"]].isna().sum()
missing
```

跑出嚟會見到：

```text
PoolQC         1453
Alley          1369
LotFrontage     259
Street            0
dtype: int64
```

**點睇：** 呢條 Series 嘅 index 係 **train 嘅欄名**（PoolQC、Alley……），值係每欄缺咗幾多格。Part 1 嘅 `missing` 表就係咁樣嚟。

```python
print(missing.index.tolist())
print(missing.values)
```

跑出嚟會見到：

```text
['PoolQC', 'Alley', 'LotFrontage', 'Street']
[1453 1369  259    0]
```

**點睇：** `.index` 攞出行名；`.values` 攞出入面嘅數值，變成一個 numpy array（冇咗行名，純粹一串數）。

**NaN：缺失值**

```python
print(small["GarageType"].tolist())
print(small["GarageType"].isna().tolist())
```

跑出嚟會見到：

```text
['Attchd', 'Attchd', 'Attchd', 'Detchd', 'Attchd']
[False, False, False, False, False]
```

**點睇：** 呢 5 間屋都有車房，所以冇缺失。如果有，pandas 會顯示 **NaN**（Not a Number），即係「缺失」嘅特別值。NaN 有兩個特性要記：NaN 加任何數都係 NaN；`NaN == 0` 係 False。

**`set_index` 同 `reset_index`**

```python
scores = pd.DataFrame({"step": ["1", "2", "5"], "cv_rmse": [0.155, 0.152, 0.132]})
by_step = scores.set_index("step").cv_rmse
print(by_step["5"])
```

跑出嚟會見到：

```text
0.132
```

**點睇：** `set_index("step")` 將 step 欄變成行名，之後就可以用 `by_step["5"]` 直接查 step 5 嘅分數。`reset_index(drop=True)` 就係反過嚟，將行名重新編成 0, 1, 2…。

### §3.2 揀行揀欄

```python
print(small[["GrLivArea", "SalePrice"]].shape)
print(small.iloc[[0, 2]]["SalePrice"].tolist())
print(small.loc[small.SalePrice > 200000, "Id"].tolist())
```

跑出嚟會見到：

```text
(5, 2)
[208500, 223500]
[1, 3, 5]
```

**點睇：**
- `small[["GrLivArea", "SalePrice"]]`：攞兩欄（**兩層**方括號：外層係「揀欄」，內層係一個 list）。`.shape` 係（行數, 欄數）= (5, 2)。
- `.iloc[[0, 2]]`：用**位置**揀第 0 同第 2 行。
- `.loc[條件, "Id"]`：揀「SalePrice 大過 20 萬」嘅行嘅 Id 欄。睇返上面張表，Id 1（208,500）、Id 3（223,500）、Id 5（250,000）超過 20 萬。

**用 `.loc` 改值：**

```python
demo = small.copy()
demo.loc[demo.GrLivArea > 2000, "GarageType"] = "BigHouseGarage"
demo[["GrLivArea", "GarageType"]]
```

跑出嚟會見到：

```text
   GrLivArea      GarageType
0       1710          Attchd
1       1262          Attchd
2       1786          Attchd
3       1717          Detchd
4       2198  BigHouseGarage
```

**點睇：** 只有 GrLivArea 大過 2000 嘅嗰一行（index 4，2198 呎）嘅 GarageType 被改咗，其他行唔郁。Part 3 用同一個寫法，將「GarageArea 係 0 而且 GarageType 缺失」嘅行改做 "NoGarage"。

`df.drop(columns="Id")`：拎走 Id 欄，回傳一張新表，原本嗰張唔變。

### §3.3 常用 method

```python
s = pd.Series([3, 0, 1, 0, 3], index=["a", "b", "c", "d", "e"])
print(s.eq(0).tolist())
print(s.isin([1, 3]).tolist())
print(s.sort_values(ascending=False).index.tolist())
print(s.value_counts().to_dict())
print(s.diff().tolist())
```

跑出嚟會見到：

```text
[False, True, False, True, False]
[True, False, True, False, True]
['a', 'e', 'c', 'b', 'd']
{3: 2, 0: 2, 1: 1}
[nan, -3.0, 1.0, -1.0, 3.0]
```

**逐行點睇：**
- `.eq(0)` 即係 `== 0`，逐格問「係咪 0？」。
- `.isin([1, 3])`：逐格問「喺唔喺呢張清單入面？」。
- `.sort_values(ascending=False)`：由大到細排（`ascending=True` 就係由細到大）。排完之後 index 嘅次序都跟住變。
- `.value_counts()`：每個值出現幾多次：3 出現 2 次、0 出現 2 次、1 出現 1 次。
- `.diff()`：每個同上一個嘅差。第一個冇上一個，所以係 NaN（`nan`）。

其他會用到嘅：

| Method | 做咩 |
|---|---|
| `.shape` | (行數, 欄數) |
| `.head(n)` / `.tail(n)` | 頭 n 行 / 最後 n 行（預設 5） |
| `.isna()` | 每格係咪缺失 |
| `.fillna(v)` | 將缺失填做 v |
| `.astype(str)` | 轉做文字 |
| `.sum()` / `.sum(axis=1)` | 每欄直向加總 / 每行橫向加總 |
| `.skew()` | 偏度（§5.3） |
| `.corr()` | 相關係數 |
| `.copy()` | 整一個獨立副本（§3.4） |
| `.clip(lower=0)` | 細過 0 嘅當 0 |
| `.round(n)` | 取 n 個小數位 |
| `.tolist()` | 變成普通 list |
| `.select_dtypes("number")` | 只揀數字欄（`exclude=` 就係排除） |
| `.to_csv(路徑, index=False)` | 存成 CSV；`index=False` 即係唔好將行名寫入檔案（否則 Kaggle 會見到多咗一欄） |
| `pd.concat([表1, 表2])` | 將兩張表上下疊埋 |
| `.empty` | 張表係咪完全冇行 |
| `.all()` | 係咪全部都係 True |
| `np.isfinite(x)` | 每個數係咪正常數字（唔係 NaN、唔係無限大） |

### §3.4 點解 function 第一行要寫 `df = df.copy()`

```python
def bad_fill(df):
    df["GarageType"] = "CHANGED"
    return df

def good_fill(df):
    df = df.copy()
    df["GarageType"] = "CHANGED"
    return df

original = small.copy()
good_fill(original)
print("after good_fill:", original.GarageType.iloc[0])
bad_fill(original)
print("after bad_fill: ", original.GarageType.iloc[0])
```

跑出嚟會見到：

```text
after good_fill: Attchd
after bad_fill:  CHANGED
```

**點睇：** `bad_fill` 直接改傳入嚟嘅表，結果**外面嗰張 `original` 都被改埋**。`good_fill` 先 `.copy()` 一份，只改副本，外面完全唔受影響。所以 Part 3 每個 function 第一行都係 `df = df.copy()`：咁 `X` 同 `X3` 先可以同時存在，你重跑任何一格都唔會撈亂。

---

## §4 Machine learning 基本概念

### §4.1 X 同 y

- **X（features，特徵）**：用嚟估嘅資料，例如面積、年份、地區。每行一間屋，每欄一個特徵。
- **y（target，目標）**：要估嘅答案，即係 SalePrice。
- **Regression（回歸）**：y 係一個數字。如果 y 係類別（例如「貴 / 平」），就叫 classification。

### §4.2 Model、`fit`、`predict`

**一句講晒：** model 係一條「由 X 計出 y」嘅規則，入面有啲數字要由數據學返嚟。`fit` 就係學，`predict` 就係用。

```python
from sklearn.linear_model import LinearRegression
X_small = train[["GrLivArea"]].head(200)
y_small = train["SalePrice"].head(200)
model = LinearRegression()
model.fit(X_small, y_small)
new_houses = pd.DataFrame({"GrLivArea": [1000, 2000]})
model.predict(new_houses).round(0)
```

跑出嚟會見到：

```text
array([122764., 234710.])
```

**逐行點睇：**
- `X_small`：200 間屋嘅居住面積（留意 `[["GrLivArea"]]` 兩層括號，因為 X 要係一張**表**，唔係一條 Series）。
- `y_small`：佢哋嘅樓價。
- `model.fit(X, y)`：學。model 睇住 200 間屋嘅面積同價錢，搵出最好嘅規則。
- `model.predict(...)`：用。俾兩間新屋（1000 呎、2000 呎），佢估出兩個價錢：大約 12.3 萬同 23.5 萬。
- 輸出嘅 `array([...])` 即係一個 numpy array（§2.1），`122764.` 後面嗰點代表小數類型。
- sklearn 入面**所有** model 都係用 `fit` 同 `predict` 呢兩個 method，所以換 model 唔使改其他 code。

### §4.3 Linear regression 同 coefficient

```python
print("b0 (intercept):", round(model.intercept_))
print("b1 (per sq ft):", round(model.coef_[0], 1))
```

跑出嚟會見到：

```text
b0 (intercept): 10817
b1 (per sq ft): 111.9
```

**點睇：** 上面嗰個 model 學到嘅規則係：

```
估計價錢 = b0 + b1 × 面積
```

- `b1` 叫 **coefficient（係數）**，即係「面積每多 1 呎，估價多幾多」。
- `b0` 叫 intercept（截距）。
- 用返上面嘅結果驗算：1000 呎 → b0 + b1 × 1000，就係 §4.2 第一個預測。
- 有好多個特徵時就係 `b0 + b1 × 特徵1 + b2 × 特徵2 + …`，每個特徵一個 coefficient。
- `.coef_`、`.intercept_` 呢類**後面有 `_`** 嘅屬性，係 sklearn 嘅慣例，代表「fit 完先有」。

**限制：** 線性 model 只可以將特徵「乘一個數再加埋」。佢唔識自己諗出「面積越大，每呎價錢越低」呢種彎嘅關係，所以要靠我哋喺 feature engineering 度幫佢（Part 3）。

### §4.4 Overfitting、underfitting、bias、variance

- **Overfitting（過度擬合）**：model 死記咗訓練數據，連啲隨機雜訊都記埋，所以訓練數據估得好準，新數據就估得差。好似背晒舊試卷答案，一換題目就唔識。
- **Underfitting（擬合不足）**：model 太簡單，連訓練數據都估唔準。
- CV（Part 2）係量度「對**新**數據有幾準」，所以 CV 可以捉到 overfitting。

**用數字感受 overfitting：** 用一棵唔限深度嘅決策樹（佢可以一直分，分到每片葉得一間屋）：

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
X_two = train[["GrLivArea", "OverallQual"]]
y_log_all = np.log(train["SalePrice"])
deep_tree = DecisionTreeRegressor(random_state=0).fit(X_two, y_log_all)
train_rmse = np.sqrt(np.mean((y_log_all - deep_tree.predict(X_two)) ** 2))
cv_rmse = -cross_val_score(DecisionTreeRegressor(random_state=0), X_two, y_log_all,
                           cv=5, scoring="neg_root_mean_squared_error").mean()
print(f"error on houses it trained on: {train_rmse:.3f}")
print(f"error on unseen houses (CV):   {cv_rmse:.3f}")
```

跑出嚟會見到：

```text
error on houses it trained on: 0.067
error on unseen houses (CV):   0.253
```

**點睇：** 對住**訓練過**嘅屋，誤差好細；對住**未見過**嘅屋（CV），誤差大好多。兩個數相差咁遠，就係 overfitting 嘅標誌。

**兩種錯誤嘅來源：**

| | 意思 | 例子 | 對應 |
|---|---|---|---|
| **Bias（偏差）** | model 太簡單，次次都錯同一個方向 | 用一條直線去 fit 一條彎線 | underfitting |
| **Variance（變異）** | model 太跟訓練數據，換一份數據結果就大變 | 上面嗰棵唔限深度嘅樹 | overfitting |

**Bias-variance trade-off：** model 越複雜，bias 越細，但 variance 越大，所以要搵中間最平衡嘅位置。Part 3 調 alpha 嘅 U 形曲線就係例子。

### §4.5 Parameter、hyperparameter、regularization

| | 由邊個決定 | 例子 |
|---|---|---|
| **Parameter** | model 喺 `fit` 時自己學 | §4.3 嘅 b0、b1 |
| **Hyperparameter** | 我哋喺 `fit` 之前設定 | Ridge 嘅 `alpha`、樹嘅深度 |

揀 hyperparameter 嘅方法：試幾個值，用 CV 比較，揀最好嗰個。呢個過程叫 **tuning（調參）**。

**Regularization（正則化）：** 喺「誤差」之外再加一項 **penalty（懲罰）**，逼 model 唔好太複雜，即係減少 variance、防止 overfitting。

```
普通線性回歸：盡量細 → 誤差² 總和
Ridge：       盡量細 → 誤差² 總和 + alpha × (所有係數²) 總和
```

`alpha` 就係正則化嘅強度：alpha 越大，懲罰越重，係數越細，model 越簡單。筆記入面「penalty」、「懲罰」、「正則化」講嘅都係呢樣嘢。

### §4.6 統計小工具、Transformer 同 Pipeline

**Median、standard deviation、標準化**

```python
areas = np.array([1000, 1500, 2000, 1200, 5600])
print("mean:  ", areas.mean())
print("median:", np.median(areas))
print("std:   ", areas.std().round(1))
print("standardized:", ((areas - areas.mean()) / areas.std()).round(2))
```

跑出嚟會見到：

```text
mean:   2260.0
median: 1500.0
std:    1703.6
standardized: [-0.74 -0.45 -0.15 -0.62  1.96]
```

**點睇：**
- **Mean（平均）** 係 2260，被 5600 呢間大屋拉高咗。
- **Median（中位數）**：由細到大排好，正中間嗰個，係 1500。佢唔會被極端值拉走，所以 Part 2 用 median 填缺失。
- **Standard deviation（標準差）**：數據平均離平均值有幾遠，即係「散得幾開」。
- **Standardization（標準化）**：`(x − 平均) ÷ 標準差`。做完之後，平均變 0、標準差變 1。5600 變成大約 1.96，意思係「高過平均差唔多 2 個標準差」。所有特徵都咁做，就全部喺同一把尺上。

**Transformer：** 負責改數據，唔負責估價。例如 `SimpleImputer`（填缺失）、`StandardScaler`（標準化）、`OneHotEncoder`。佢哋用 `fit` 學嘢（例如學 median 係幾多），再用 `transform` 套用。

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(pd.DataFrame({"LotFrontage": [60, 80, np.nan, 70]}))
print("learned median:", imputer.statistics_)
imputer.transform(pd.DataFrame({"LotFrontage": [np.nan, 90]}))
```

跑出嚟會見到：

```text
learned median: [70.]
array([[70.],
       [90.]])
```

**點睇：** `fit` 學到 median 係 70（由 60、70、80 計出嚟，NaN 唔計）。`transform` 將新數據嘅 NaN 填做 70，90 就保持不變。輸出 `[[70.], [90.]]` 係一張 2 行 1 欄嘅表：每個 `[ ]` 係一行。**關鍵：** 佢只係記住 `fit` 嗰陣學到嘅 70。所以如果 `fit` 只用訓練數據，驗證數據就完全冇份影響呢個 70，即係冇 leakage。

**Pipeline：** 將幾個 transformer 同最後嘅 model 串成一條線。
- `pipeline.fit(X, y)`：逐個 transformer 喺**同一份訓練數據**上 fit 同 transform，最後 fit model。
- `pipeline.predict(X_new)`：逐個 transform（用 fit 時學返嚟嘅數字），最後 predict。
- 好處：成條 pipeline 可以當成一個 model 用，自動避開 leakage（Part 2 §1.3）。

### §4.7 RMSE

```python
true_log = np.log(np.array([200000, 150000, 300000]))
pred_log = np.log(np.array([220000, 140000, 300000]))
errors = true_log - pred_log
print("errors:", errors.round(3))
print("RMSE:  ", np.sqrt(np.mean(errors ** 2)).round(4))
```

跑出嚟會見到：

```text
errors: [-0.095  0.069  0.   ]
RMSE:   0.0679
```

**點睇：**
1. 計每間屋嘅誤差（真實 − 估計）：第一間估高咗 10%，誤差大約 −0.095；第二間估低咗 7%，誤差大約 +0.069；第三間啱晒，誤差 0。
2. 平方：令負數變正，大錯罰得更重。
3. 平均。
4. 開方：變返原本嘅單位。

得出嘅 RMSE 就係 Kaggle 計分用嘅數字。**越細越好。**

---

## §5 Log 嘅數學直覺

### §5.1 Log 係咩

`log(x)` 回答一個問題：「e（約 2.718）要自乘幾多次先等於 x？」你**唔使識計**，只要記住佢嘅**性質**（§5.2）。

```python
for value in [1, 10, 100, 1000, 180000]:
    print(f"log({value}) = {np.log(value):.2f}")
```

跑出嚟會見到：

```text
log(1) = 0.00
log(10) = 2.30
log(100) = 4.61
log(1000) = 6.91
log(180000) = 12.10
```

**點睇：** 數字每大 10 倍，log 只係加大約 2.3。所以 log 會將一大堆好大嘅數「壓扁」：18 萬變成 12.1。

### §5.2 最重要嘅性質：乘數變加數

```
log(a × b) = log(a) + log(b)
```

```python
for price in [100000, 1000000]:
    gap = np.log(price * 1.1) - np.log(price)
    print(f"{price:>8} -> {price * 1.1:>9.0f}: log goes up by {gap:.3f}")
```

跑出嚟會見到：

```text
  100000 ->    110000: log goes up by 0.095
 1000000 ->   1100000: log goes up by 0.095
```

**點睇：** 貴 10% 喺 log 尺度上永遠係 +0.095，無論係 10 萬嘅屋定 100 萬嘅屋。所以喺 log 尺度，「貴 10%」對平樓同貴樓都係同一個距離。呢個就係 Kaggle 用 log 計分、同埋我哋預測 log 價錢嘅原因。

### §5.3 Skewness（偏度）同 log 點樣「拉直」數據

```python
print("SalePrice skew:      ", round(train.SalePrice.skew(), 2))
print("log(SalePrice) skew: ", round(np.log(train.SalePrice).skew(), 2))
print("LotArea skew:        ", round(train.LotArea.skew(), 2))
print("log1p(LotArea) skew: ", round(np.log1p(train.LotArea).skew(), 2))
```

跑出嚟會見到：

```text
SalePrice skew:       1.88
log(SalePrice) skew:  0.12
LotArea skew:         12.21
log1p(LotArea) skew:  -0.14
```

**點睇：**
- **Skew = 0** 代表左右對稱。
- **正數**代表有條長尾巴拖向右邊，即係大部份數值細，少數極大（例如面積、樓價）。
- 慣例：skew > 0.75 就當「明顯偏」，要考慮取 log；> 1 就係好偏。0.75 唔係定律，只係常用門檻。
- 取 log 之後，skew 大幅跌向 0，條尾巴縮短咗，分佈接近對稱。原因係 §5.1：log 壓扁大數多過細數。

### §5.4 `log1p`

```python
print(np.log1p(np.array([0, 1, 100, 10000])).round(3))
print(np.log(np.array([1, 2, 101, 10001])).round(3))
```

跑出嚟會見到：

```text
[0.    0.693 4.615 9.21 ]
[0.    0.693 4.615 9.21 ]
```

**點睇：**
- `log1p(x) = log(1 + x)`，所以兩行結果一模一樣。
- 好處：x = 0 都計得（得 0），而 `log(0)` 係負無限大，會爆。面積可以係 0（例如冇泳池、冇露台），所以我哋對特徵用 log1p。
- 數字大嘅時候，log1p 同 log 幾乎一樣。

---

## §6 Matplotlib（`plt`）：畫圖

**Figure 同 Axes：** `fig` 係成張紙，`ax` 係紙上面其中一格圖。一張紙可以有幾格（叫 `axes`）。畫嘢係對 `ax` 做，存檔係對 `fig` 做。

```python
fig, ax = plt.subplots(figsize=(7, 5))       # 一張紙、一格圖，闊 7 吋高 5 吋
ax.scatter(train.GrLivArea, train.SalePrice)  # 喺格圖度畫散點
ax.set(title="Living area vs price", xlabel="GrLivArea", ylabel="SalePrice")
fig.tight_layout()                            # 自動調整，費事啲字重疊
fig.savefig("figure.png", dpi=120)            # 存檔（dpi 越大越清）
plt.show()                                    # 喺 notebook 顯示
```

| 寫法 | 意思 |
|---|---|
| `fig, axes = plt.subplots(1, 2)` | 1 行 2 格，`axes[0]`、`axes[1]` 係兩格圖 |
| `ax.hist(...)` | 直方圖：數據分佈 |
| `ax.bar(...)` / `ax.barh(...)` | 直棒圖 / 橫棒圖（橫棒圖由下而上畫） |
| `ax.scatter(...)` | 散點圖 |
| `ax.imshow(表)` | 將一張數字表畫成一格格顏色（熱力圖） |
| `ax.set(title=, xlabel=, ylabel=, xlim=, ylim=)` | 一次過設標題、軸名、軸範圍 |
| `ax.text(x, y, "字", ha="center", va="center")` | 喺 (x, y) 寫字；`ha` / `va` 係水平 / 垂直對齊方式 |
| `ax.legend()` | 顯示圖例 |

每幅圖真正嘅樣，可以喺 Part 1–5 嘅筆記入面直接睇到。
