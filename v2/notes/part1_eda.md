# Part 1 學習筆記：EDA（探索數據）

> EDA 唔係用嚟攞分。佢做嘅係**搵出之後每一步要做咩，同埋點解要咁做**。
> Part 3 每一個 feature engineering step，都可以喺呢度搵到佢嘅源頭，所以報告講「pathway」嗰陣，EDA 就係起點。

---

## 1. 概念

### 1.1 呢個比賽考咩

- 有 1460 間知道樓價嘅屋，叫 **train**；另外有 1459 間唔知樓價嘅屋，叫 **test**。要估 test 嗰批嘅樓價。
- 每間屋有 79 個描述欄，例如面積、年份、質素、地段。
- 呢個係 **regression**，因為要估嘅係一個數字，唔係一個類別。

### 1.2 Kaggle 點樣計分：log RMSE

$$\text{score} = \sqrt{\frac{1}{n}\sum_i \big(\log(\hat y_i) - \log(y_i)\big)^2}$$

直覺上，log 嘅差其實係**百分比誤差**：

- 一間 10 萬嘅屋估成 11 萬，即係估多 10%。
- 一間 100 萬嘅屋估成 110 萬，都係估多 10%。

兩個喺 log 尺度上嘅誤差幾乎一樣。如果唔用 log，第二個錯咗 10 萬，第一個只係錯咗 1 萬，平樓嘅誤差就會被忽略。**結論：分數只理比例，唔理絕對金額。**

### 1.3 Skewness（偏度）

- 數據有一條長尾巴拖向右邊（少數極貴嘅屋），就叫 **right-skewed**。
- skew = 0 代表左右對稱；> 1 就算好偏。
- 我哋嘅 SalePrice skew = **1.88**，取 log 之後得 **0.12**，差唔多完全對稱。

**點解偏斜會影響線性 model？** 線性回歸要減嘅係誤差嘅平方。尾巴嗰幾間貴屋誤差特別大，一平方就更加大，結果成條線都被佢哋拉走。

### 1.4 缺失值有兩種

| 種類 | 例子 | 意思 | 應該點處理 |
|---|---|---|---|
| **NA = 冇呢樣嘢** | PoolQC、GarageType、Fence | 間屋冇泳池、冇車房 | 填一個明確嘅「冇」標籤（Part 3 Step 3） |
| **NA = 唔知** | LotFrontage | 數據收集時漏咗 | 用 median 之類嘅估值填（Part 2 Pipeline） |

呢個分別係從 `data_description.txt` 讀返嚟嘅。**睇數據說明書都係 feature engineering 嘅一部份**，報告值得提。

### 1.5 Outlier（離群值）

有兩間屋大過 4000 平方呎，但只係賣十幾萬。兩間都係 `Partial` sale，即係新樓未起好就賣咗，所以唔跟正常嘅價錢規律。如果 model 硬要遷就佢哋，條線就會整歪，其他 1458 間屋都會受影響。

**重要：** 我哋只係喺 **training** 嗰陣拎走佢哋，計 CV 分數嗰陣佢哋照計。如果連評分都拎走，就變咗呃自己，因為 test 入面可能都有類似嘅屋。呢點 Part 3 會再講。

### 1.6 Correlation（相關系數）

- 數值由 -1 到 1，量度兩個數字**有幾似一條直線咁一齊郁**。
- 局限一：只睇得到直線關係。
- 局限二：睇唔到文字欄（我哋有 43 個文字欄）。
- 所以佢只係一張「快速地圖」，唔應該用嚟決定刪走邊啲 feature。

---

## 2. 逐行解釋

### Cell 1：Setup 同讀數據

```python
from pathlib import Path
```
`Path` 係處理檔案路徑嘅工具。`Path("a") / "b.csv"` 會自動砌成 `a/b.csv`，Windows、Mac、Linux 都通用。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
- numpy (`np`) 做數學運算，例如 `np.log`。
- pandas (`pd`) 處理表格（DataFrame）。
- matplotlib (`plt`) 畫圖。

`np`、`pd`、`plt` 呢三個縮寫係業界慣例，人人都咁寫。

```python
CANDIDATES = [Path("/kaggle/input/house-prices-advanced-regression-techniques"),
              Path("../house-prices-advanced-regression-techniques"),
              Path("house-prices-advanced-regression-techniques")]
DATA_DIR = next(path for path in CANDIDATES if (path / "train.csv").exists())
```
- 喺 Kaggle 上面，data 固定放喺 `/kaggle/input/...`；喺我嗰邊（本地）就係另一個位置。
- 呢兩行會逐個位置去搵，邊度搵到 `train.csv` 就用邊個。
- `next(...)` 嘅意思係「攞第一個符合條件嘅」。
- **冇咗會點：** 同一本 notebook 唔可以喺 Kaggle 同本地都跑得到。

```python
OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("results")
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
```
- Kaggle 只准寫檔案入 `/kaggle/working`，所以 output 要放嗰度。
- `mkdir` 開 folder。`parents=True` 代表連上層 folder 一齊開；`exist_ok=True` 代表已經有就唔好報錯。

```python
def read_csv(name):
    return pd.read_csv(DATA_DIR / name, keep_default_na=False, na_values=["NA", ""])
```
**呢行係成個 Part 最容易中伏嘅地方。**

- pandas 預設會將 `"None"` 呢個字當成缺失值。
- 但喺呢份數據入面，`MasVnrType = "None"` 意思係「冇石材外牆」，係一個真嘅類別（有 864 間屋係咁）。
- `keep_default_na=False` 叫 pandas 唔好用佢預設嗰張「當成缺失」嘅清單。
- `na_values=["NA", ""]` 就話佢知，只有 `NA` 同空格先算缺失。
- **冇咗會點：** 864 個真實資料會被當成「唔知」，間接整壞 model。

```python
train = read_csv("train.csv")
test = read_csv("test.csv")
sample = read_csv("sample_submission.csv")
print("train:", train.shape, " test:", test.shape)
```
- 讀三個檔案。`sample` 係 Kaggle 俾嘅提交格式範本，之後用嚟對齊 Id。
- `.shape` 回傳（行數, 欄數）。結果係 train (1460, 81)，test (1459, 80)。test 少一欄，因為冇 SalePrice。

### Cell 2：First look

```python
print(train.dtypes.value_counts())
train.head()
```
- 數下每種資料類型有幾多欄：object（文字）43 欄、int64 35 欄、float64 3 欄。
- `head()` 顯示頭 5 行。
- 一個 cell 最後嗰行嘅結果，Jupyter 會自動顯示出嚟，所以唔使寫 `print`。

### Cell 3：SalePrice 分佈圖

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
```
開一幅圖，入面有 1 行 2 格（`axes[0]` 同 `axes[1]`），闊 11 吋、高 4 吋。

```python
axes[0].hist(train.SalePrice, bins=50, color="steelblue")
axes[0].set(title=f"SalePrice (skew = {train.SalePrice.skew():.2f})", xlabel="price ($)")
```
- `hist` 畫直方圖，`bins=50` 即係將價錢切成 50 段。
- `f"...{...:.2f}"` 係 f-string，會將計出嚟嘅 skew 塞入標題，`:.2f` 代表顯示 2 個小數位。

```python
axes[1].hist(np.log(train.SalePrice), bins=50, color="seagreen")
```
同一樣嘢，不過先取 log 再畫。兩幅並排放，一眼就睇到取 log 前後嘅分別。

```python
fig.tight_layout()
fig.savefig(OUT_DIR / "figures" / "01_saleprice_distribution.png", dpi=120)
plt.show()
```
- `tight_layout` 自動調整，費事啲字重疊。
- `savefig` 存圖，之後放入報告同 slides。
- `show` 喺 notebook 入面顯示幅圖。

### Cell 4：缺失值

```python
ABSENT_MEANING = ["PoolQC", "MiscFeature", ...]
```
人手列出「NA = 冇」嘅欄。呢張清單係**睇 `data_description.txt` 得返嚟嘅**，唔係計出嚟。

```python
missing = pd.DataFrame({"train": train.drop(columns="SalePrice").isna().sum(),
                        "test": test.isna().sum()})
```
- `isna()` 將每一格變成 True（缺失）或者 False。
- `.sum()` 將每欄嘅 True 數埋，即係每欄缺幾多格。
- train 同 test 分開兩欄數，方便比較。

```python
missing = missing[(missing.train + missing.test) > 0]
```
只留低有缺失嘅欄。

```python
missing["kind"] = np.where(missing.index.isin(ABSENT_MEANING), "NA = absent", "truly unknown")
```
`np.where(條件, A, B)`：條件成立就填 A，唔成立就填 B。用嚟幫每欄標記係邊種缺失。

```python
print("Columns missing only in test:", missing.index[missing.train.eq(0)].tolist())
```
- 搵出 train 冇缺、但 test 有缺嘅欄。結果有 15 欄，例如 `MSZoning`、`GarageCars`。
- **點解重要：** model 喺 training 時從未見過呢啲欄有缺失。如果唔處理，到 test 先會爆。Part 2 嘅 Pipeline 會負責填佢哋。

```python
top = missing.head(20).iloc[::-1]
colors = np.where(top.kind.eq("NA = absent"), "orange", "gray")
ax.barh(top.index, top.train, color=colors)
```
- 揀缺失最多嘅 20 欄。`iloc[::-1]` 將次序倒轉，咁最多嗰欄就會喺橫向棒形圖嘅最上面。
- 橙色代表「NA = 冇」，灰色代表「唔知」。

### Cell 5：Outlier

```python
outlier = (train.GrLivArea > 4000) & (train.SalePrice < 300000)
```
- 兩個條件同時成立先算 outlier。`&` 係「同埋」；pandas 要用 `&`，唔可以用 `and`，而且每個條件要加括號。
- **點解唔係只用 `> 4000`？** 因為有兩間大屋賣到 70 幾萬，佢哋完全跟正常規律（大屋貴價），唔應該拎走。

```python
ax.scatter(train.GrLivArea, train.SalePrice, s=8, alpha=0.5)
ax.scatter(train.GrLivArea[outlier], train.SalePrice[outlier], s=150,
           facecolors="none", edgecolors="red", linewidths=2, label="outliers")
```
- 第一行畫全部點。`alpha=0.5` 係半透明，點重疊嘅地方會深色啲。
- 第二行喺 outlier 上面再畫一個空心紅圈。

```python
train.loc[outlier, ["Id", "GrLivArea", "SalePrice", "OverallQual", "SaleCondition"]]
```
- 用 `.loc[邊啲行, 邊啲欄]` 攞出嗰兩間屋嘅資料。
- 會見到 `SaleCondition = Partial`，呢個就係拎走佢哋嘅**理由**，唔係淨係因為佢哋「睇落怪」。

### Cell 6：相關系數

```python
correlations = (train.select_dtypes("number").drop(columns="Id")
                .corr()["SalePrice"].drop("SalePrice").sort_values())
```
由左至右讀：

1. 只揀數字欄；
2. 拎走 Id，因為 Id 只係編號，冇意思；
3. `.corr()` 計出所有欄兩兩之間嘅相關；
4. 只攞同 SalePrice 嗰一行；
5. 拎走 SalePrice 自己（自己同自己一定係 1）；
6. 由細到大排好。

---

## 3. 點睇結果

| 結果 | 代表咩 |
|---|---|
| SalePrice skew 1.88 → log 後 0.12 | 預測 log 價錢會令線性 model 舒服好多（Step 2） |
| PoolQC 缺 1453/1460 | 唔係數據差，而係 99.5% 嘅屋都冇泳池 |
| LotFrontage 缺 259 | 真係唔知，要估值填返 |
| 15 欄只喺 test 有缺失 | Pipeline 一定要識處理「未見過嘅缺失」 |
| 2 個 outlier（Id 524、1299） | Step 6 測試拎走佢哋 |
| OverallQual 0.79、GrLivArea 0.71 | 質素同面積主導價錢，所以 Step 4 要整「總面積」 |

最尾嗰個 markdown 表（「What EDA tells us to do next」）將每個發現連去之後嘅 step。**報告講 pathway 嗰陣，可以直接用呢張表做骨幹。**

---

## 4. 講師可能會問

**Q：點解要預測 log 價錢，唔直接預測價錢？**
A：有兩個原因。第一，Kaggle 本身就係喺 log 尺度計分，喺同一個尺度訓練先最直接。第二，樓價係 right-skewed（1.88），取 log 之後接近對稱（0.12），線性 model 就唔會被幾間貴屋拉走。Step 2 會用 CV 分數證明呢點。

**Q：點解唔將缺失太多嘅欄（例如 PoolQC）直接刪走？**
A：因為 PoolQC 嘅 NA 唔係「唔知」，而係「冇泳池」，本身就係資訊。刪走就會連「有冇泳池」呢個資訊一齊冇咗。正確做法係將 NA 改做一個明確嘅「NoPool」標籤。

**Q：拎走 outlier 算唔算作弊？**
A：唔算，前提係三點：(1) 有理由，嗰兩間係 Partial sale，唔跟正常市場規律；(2) 只喺 training 拎走，計分時照計；(3) 用 CV 證明拎走之後真係有幫助（Step 6）。如果連評分都拎走，先係呃自己。

**Q：Correlation 低嘅欄係咪可以刪走？**
A：唔應該。Correlation 只睇到直線關係，亦睇唔到文字欄。例如 Neighborhood（地區）好影響樓價，但佢係文字欄，correlation 根本計唔到。

**Q：點解讀 CSV 要加 `keep_default_na=False`？**
A：因為 pandas 預設會將 `"None"` 當成缺失，但喺呢份數據入面，`"None"` 係一個真實類別（冇石材外牆，有 864 間屋）。唔加呢個設定，就會誤刪真實資訊。

---

## 有用嘅課堂連結（你有課程目錄之後我再補）

- Data preprocessing：缺失值處理、outlier
- Data exploration：分佈、skewness、correlation
