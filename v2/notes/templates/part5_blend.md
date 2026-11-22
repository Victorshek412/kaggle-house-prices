# Part 5 學習筆記：Blending 同成條 pathway 嘅總結

```python setup
import numpy as np
```


> **前置知識（唔識就先睇 Part 0）：** dict 同 list → §1.2；for 迴圈（兩層）→ §1.4；numpy 平均 → §2.2；DataFrame → §3。
> 另外會用到 Part 4 嘅 `oof`、`test_preds`、`names`、`y_log`、`comparison`，同 Part 2 嘅 `record`、`save_submission`、`results`。

---

## 0. 成個 Part 嘅思路同結果

### 0.1 思路

Part 4 揀到最好嘅單一 model 係 XGBoost（0.1231）。Part 5 問嘅問題係：**將幾個 model 嘅預測平均，會唔會好過最好嗰一個？**

答案係會，但有條件：**幾個 model 要錯得唔一樣。**

### 0.2 結果

| | CV RMSE |
|---|---|
| 最好嘅單一 model（XGBoost） | 0.1231 |
| **Step 12 blend（Lasso + XGBoost + CatBoost + SVR）** | **0.1191** |
| 進步（同 XGBoost 比） | −0.0040（3.3%） |

**小心：** notebook 印出嚟嘅係「Step 12 … (−0.00442 vs previous step)」，因為 `record` 係同成績表上一行（SVR）比較，唔係同最好嘅 model 比（Part 4 §0.3 講過呢個陷阱）。報告要用 −0.0040。

---

## 1. 概念：點解平均可以好過最好嗰個

### 1.1 一個簡單例子

一間屋真實 log 價錢係 12.00：

| | 估計 | 錯誤 |
|---|---|---|
| Model A | 12.10 | +0.10（估高咗） |
| Model B | 11.94 | −0.06（估低咗） |
| A、B 平均 | 12.02 | **+0.02** |

A 估高、B 估低，平均之後兩邊互相抵消，所以比兩個都準。

但如果 A、B **錯嘅方向一樣**，例如都估高 0.10，平均之後仍然係錯 0.10，冇任何改善。

### 1.2 點樣量度「錯得一唔一樣」：residual correlation

- **Residual（殘差）** = 真實 log 價錢 − 估計 log 價錢，即係每間屋錯咗幾多（有正負）。
- 每個 model 都有 1460 個 OOF residual（Part 2 講過 OOF 係咩）。
- 計兩個 model 嘅 residual 之間嘅 correlation（Part 1 §1.6 講過 correlation 係咩）：
  - **1.0**：每間屋都錯得一模一樣，blend 冇用；
  - **越低**：錯得越唔同，blend 越有用。

> 呢個就係 **diversity（多樣性）** 嘅概念：ensemble 嘅成員要「各有所長、各有所短」。成班一模一樣嘅專家，等於得一個專家。

**用 4 間屋證明「錯得唔同」先有用：**

```python run
true = np.array([12.0, 11.5, 12.3, 11.8])
a = true + np.array([0.10, -0.05, 0.08, -0.02])   # model A 嘅錯
b = true + np.array([0.09, -0.04, 0.07, -0.03])   # model B：錯法同 A 差唔多
c = true + np.array([-0.08, 0.06, -0.05, 0.03])   # model C：錯法同 A 相反
rmse = lambda prediction: np.sqrt(np.mean((true - prediction) ** 2))
print(f"A alone:  RMSE {rmse(a):.3f}")
for name, other in [("B", b), ("C", c)]:
    correlation = np.corrcoef(true - a, true - other)[0, 1]
    print(f"A + {name}: residual correlation {correlation:+.2f}, blend RMSE {rmse((a + other) / 2):.3f}")
```

**點睇：**
- A 同 B 錯法差唔多（correlation 接近 +1），平均之後誤差幾乎冇改善。
- A 同 C 錯法相反（correlation 係負數），平均之後誤差大幅跌，因為一個估高、一個估低，互相抵消。
- 所以揀 blend 成員，唔係揀「最好嘅幾個」，而係揀「好、而且錯法唔同」嘅幾個。

### 1.3 點解唔調權重（例如 XGBoost 40%、Lasso 20%……）

調權重好似好合理，但有一個陷阱：

- 權重要喺某啲數據上面揀，而我哋手上只有呢 1460 個 OOF 預測。
- 用呢啲數據揀權重，再用**同一啲**數據報分數，個分數一定虛高。權重會學埋呢 1460 間屋嘅雜訊，即係 **overfitting**，只不過今次唔係喺 model 層面，而係喺 blend 層面。
- 等權重（每個 25%）**唔使睇分數都可以決定**，所以冇呢個問題。

### 1.4 點樣揀邊幾個 model：先定規則，再睇結果

如果我試晒所有組合，再揀分數最好嗰個，就同調權重一樣，會 overfit CV。所以我哋**先定規則**：

> **每個家族揀最好嗰個；兩個 boosting 都要，因為佢哋處理類別嘅方法唔同。**

| 家族 | 候選 | 揀 | 原因 |
|---|---|---|---|
| 線性 | Ridge、Lasso、ElasticNet | **Lasso** | 三個 residual correlation 係 0.99–1.00，幾乎一樣；Lasso 分數最好 |
| Bagging | Random Forest | ✗ | 差太遠（0.138），會拖低平均 |
| Boosting | XGBoost、CatBoost | **兩個都要** | one-hot vs target statistics，correlation 0.93，唔係完全一樣 |
| Kernel | SVR | **SVR** | 思路完全唔同（相似度），分數好 |

---

## 2. 逐行解釋

### Cell 1：計 residual 同畫熱力圖

```python
residuals = pd.DataFrame({name: y_log - oof[name] for name in names})
```
- `{name: ... for name in names}` 叫 **dict comprehension**，同 Part 4 講過嘅 list comprehension 一樣，只係結果係 dict：逐個 model 名，計佢嘅 residual。
- `y_log - oof[name]`：1460 個真實值減 1460 個預測，逐個相減。
- 包成 DataFrame：每欄係一個 model，每行係一間屋。

```python
residual_corr = residuals.corr()
```
`.corr()` 會計每兩欄之間嘅 correlation，結果係一張 7×7 嘅表。

```python
fig, ax = plt.subplots(figsize=(6.5, 5.5))
image = ax.imshow(residual_corr, cmap="Blues", vmin=0.7, vmax=1)
```
- `imshow` 將一張數字表畫成一格格顏色，叫**熱力圖（heatmap）**。
- `cmap="Blues"`：數值越大，藍色越深。
- `vmin=0.7, vmax=1`：顏色範圍由 0.7 到 1。因為所有數字都喺呢個範圍，如果用 0 到 1，全部格都會差唔多深色，睇唔出分別。

```python
ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
ax.set_yticks(range(len(names)), names)
```
- 喺 x 軸同 y 軸標上 model 名。
- `range(len(names))` 即係 0, 1, …, 6，每一格嘅位置。
- `rotation=45`：x 軸啲字打斜 45 度，費事重疊。

```python
for i in range(len(names)):
    for j in range(len(names)):
        value = residual_corr.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if value > 0.93 else "black")
```
- 兩層 for 迴圈，行遍 7×7 = 49 格，每格寫上數字。
- `.iloc[i, j]`：攞第 i 行、第 j 欄嘅數字。
- `ax.text(x, y, 字)`：喺 (x, y) 寫字。留意 x 係**欄**（j），y 係**行**（i）。
- `"white" if value > 0.93 else "black"`：Python 嘅一行 if：深色格用白字，淺色格用黑字，咁先睇得清楚。

```python
fig.colorbar(image, label="correlation of OOF residuals")
```

跑出嚟嘅圖：

![OOF residual correlation](../results/figures/06_oof_correlation.png)

點睇：每一格係「行嗰個 model」同「欄嗰個 model」嘅錯誤有幾相似。對角線一定係 1.00（自己同自己）。顏色越深越似。左上角三個線性 model 深藍一片，即係錯法幾乎一樣。
喺旁邊加一條色標，話俾人知邊種顏色代表幾多。

### Cell 2（markdown）：點睇熱力圖

| 一對 model | Correlation | 意思 |
|---|---|---|
| Ridge / Lasso / ElasticNet | 0.99–1.00 | 幾乎一模一樣，互相 blend 冇用 |
| 線性 vs boosting | 0.88 | 好 model 之間最唔似嘅一對，blend 最有用 |
| XGBoost vs CatBoost | 0.93 | 同屬 boosting，但處理類別方法唔同，所以都有少少分別 |
| SVR vs 其他（RF 除外） | 0.89–0.92 | 介乎中間 |
| Random Forest vs 線性 | 0.77 | 錯得最唔同，但本身太差（§3 解釋點解都唔要） |

### Cell 3：做 blend

```python
blend_names = ["lasso", "xgb", "catboost", "svr"]
blend_oof = np.mean([oof[name] for name in blend_names], axis=0)
```
- `[oof[name] for name in blend_names]`：攞出 4 個 model 嘅 OOF 預測，組成一個 list，入面有 4 個 array，每個 1460 個數。
- `np.mean(..., axis=0)`：沿住「model」嗰個方向平均。即係每間屋都將 4 個預測加埋除 4，結果係 1460 個數。
  - `axis=0` 嘅意思：將 4 個 array 疊成一個 4×1460 嘅表，然後**直向**平均每一欄。

**用 2 個 model、3 間屋睇 `axis=0`：**

```python run
predictions = [np.array([12.1, 11.8, 12.5]),   # model A 對 3 間屋嘅預測
               np.array([11.9, 11.9, 12.3])]   # model B 對同樣 3 間屋嘅預測
print(np.mean(predictions, axis=0))
```

**點睇：** 每間屋都係兩個預測嘅平均：(12.1 + 11.9) / 2 = 12.0、(11.8 + 11.9) / 2 = 11.85、(12.5 + 12.3) / 2 = 12.4。`axis=0` 就係「沿住 model 嗰個方向平均」，即係「直向」平均。

```python
blend_test = np.mean([test_preds[name] for name in blend_names], axis=0)
```
test 預測用同一個方法平均，得出 1459 個數。

```python
blend_score = np.sqrt(np.mean((y_log - blend_oof) ** 2))
```
用 blend 嘅 OOF 預測計 RMSE，公式同 `cv_evaluate` 入面一樣。

**點解 blend 唔使再跑一次 CV？** 因為每個 model 嘅 OOF 預測本身已經係「冇見過嗰間屋」嘅預測。將冇見過嘅預測平均，結果都係冇見過嘅，所以呢個分數同樣公平。

```python
record("12", "Blend: Lasso + XGBoost + CatBoost + SVR, equal weights", blend_score)
save_submission("s6_blend", blend_test)
```

跑出嚟會見到：

```text
Step 12: CV RMSE 0.11906 (-0.00442 vs previous step)
saved s6_blend.csv
```

−0.00442 係同上一行（SVR，0.12349）比。同最好嘅 XGBoost（0.12310）比係 −0.0040，報告用後者。
記錄，出 submission **S6**。（交咗 Kaggle 之後，Part 6 再做咗一次 error analysis，出咗 S7。）

### Cell 4：其他 blend 做對比（只係參考）

```python
def blend_rmse(chosen):
    return np.sqrt(np.mean((y_log - np.mean([oof[n] for n in chosen], axis=0)) ** 2))
```
將 Cell 3 嘅計法包成 function，俾任何一組 model 都可以計 blend 分數。

```python
pd.Series({
    f"best single model ({comparison.index[0]})": comparison.cv_rmse.min(),
    "XGBoost + CatBoost": blend_rmse(["xgb", "catboost"]),
    ...
}).round(5).to_frame("cv_rmse")
```

跑出嚟會見到：

```text
                                 cv_rmse
best single model (xgb)          0.12310
XGBoost + CatBoost               0.12099
Lasso + XGBoost                  0.12044
chosen 4-model blend             0.11906
all models except Random Forest  0.12022
```
- 用 dict 整一個 Series，再用 `.to_frame` 轉做表格，咁顯示會靚啲。
- 第一行嘅名用 f-string 自動填上最好嘅 model 名（`comparison.index[0]`，即係比較表排第一嗰個），所以喺 Kaggle 跑如果排名有變，個名都會啱。
- **重要：** 呢啲數字係**揀完先睇**嘅，用嚟理解，唔係用嚟揀。

| Blend | CV RMSE | 解讀 |
|---|---|---|
| 最好單一 model（XGBoost） | 0.1231 | 起點 |
| XGBoost + CatBoost | 0.1210 | 兩個 boosting 都有少少互補 |
| Lasso + XGBoost | 0.1204 | 線性 + 樹，更互補（correlation 0.88 < 0.93） |
| **揀中嘅 4 個** | **0.1191** | 4 個家族，最互補 |
| 除 Random Forest 外全部 6 個 | 0.1202 | 反而差啲！ |

**點解「全部都用」反而差？** 6 個入面有 3 個係幾乎一樣嘅線性 model，等於線性 model 有 3 票，boosting 同 SVR 加埋先得 3 票。投票權被重複嘅 model 霸佔，平均就偏向線性 model 嘅錯誤。**數量多唔代表好，多樣性先重要。**

### Cell 5：進步樓梯圖

```python
steps = results[results.step != "0"].reset_index(drop=True)
```
- 唔畫 step 0（0.41），因為佢比其他 step（0.12–0.16）大好多，畫埋就會將其他棒壓到睇唔出分別。
- `.reset_index(drop=True)`：重新由 0 開始編行號，方便之後用 `i` 對位。

```python
colors = ["gray" if s == "1" else "orange" if s in {"2", "3", "4", "5", "6", "6b"}
          else "steelblue" if s != "12" else "seagreen" for s in steps.step]
```
- 一個 list comprehension，逐個 step 決定用咩顏色：
  - step 1 → 灰（baseline）
  - 2–6b → 橙（feature engineering）
  - 12 → 綠（blend）
  - 其餘 → 藍（model）
- `a if 條件1 else b if 條件2 else c`：連續嘅一行 if，由左至右逐個條件檢查。
- `{"2", "3", …}` 係一個 set（集合），`in` 用嚟查「係咪喺入面」。

```python
ax.bar(steps.step, steps.cv_rmse, color=colors)
for i, value in enumerate(steps.cv_rmse):
    ax.text(i, value + 0.0005, f"{value:.4f}", ha="center", fontsize=7)
```
- `bar` 畫直棒。
- `enumerate(...)` 一路行一路俾埋位置編號 `i`（0, 1, 2…）同數值 `value`。
- 喺每條棒頂上面少少（+0.0005）寫上分數。

```python
ax.set(..., ylim=(0.11, 0.16))
```

跑出嚟嘅圖：

![Improvement staircase](../results/figures/07_improvement_staircase.png)

跑出嚟會見到：

```text
Step 0 (predict the mean) scored 0.4078; it is left off the chart.
```

（呢行係 cell 最後嘅 `print` 印嘅，提醒睇圖嘅人 Step 0 冇畫入去。）
- y 軸由 0.11 開始，唔係由 0 開始。
- **報告用呢幅圖時一定要講明**：因為 y 軸唔由 0 開始，棒嘅高度差會睇落比實際大。例如 step 5 對 step 4 嘅跌幅係 12.4%，唔係幅圖睇落咁似一半。

### Cell 6：Submission 總表

```python
submission_steps = {"s1_ridge_raw": "1", "s2_new_features": "4", ...}
cv_by_step = results.set_index("step").cv_rmse
```
- 一個 dict：submission 檔名 → 對應嘅 step。
- `set_index("step")` 用 step 做行名，之後可以用 `cv_by_step["5"]` 直接查分數。

```python
summary = pd.DataFrame({
    "submission": list(submission_steps),
    "step": list(submission_steps.values()),
    "cv_rmse": [cv_by_step[s] for s in submission_steps.values()],
})
summary.to_csv(OUT_DIR / "submission_summary.csv", index=False)
```

跑出嚟會見到：

```text
        submission step  cv_rmse
0     s1_ridge_raw    1  0.15534
1  s2_new_features    4  0.15121
2  s3_log_features    5  0.13239
3   s4_no_outliers   6b  0.12766
4      s5_catboost   10  0.12320
5         s6_blend   12  0.11906
```
- `list(dict)` 會攞出所有 key；`.values()` 攞出所有 value。
- 存成 `submission_summary.csv`。
- **留意：** 另外有一個 `v2/results/kaggle_log.csv` 係俾你人手填 Kaggle 分數嘅，notebook **唔會**覆蓋佢。所以你填完之後，重跑 notebook 都唔會冇咗。

### Cell 7（markdown）：總結

呢段英文總結可以直接用喺報告嘅 conclusion。

---

## 3. 點睇結果

### 3.1 Blend 0.1191 vs 最好單一 model 0.1231

- 進步 3.3%，**代價係零**：唔使新特徵，唔使新 model，只係平均已有嘅預測。
- 證明咗 Part 4 嘅 model 之間真係有互補，同熱力圖嘅結論一致。

### 3.2 點解唔要 Random Forest，佢明明同其他 model 最唔似（0.77）

- 錯得唔似只係一個條件，另一個條件係**本身唔可以太差**。
- Random Forest 嘅誤差（0.138）比其他大好多，平均入去，佢嘅大錯會拖累整體。
- 理想嘅 blend 成員係「**好，而且唔似**」，兩樣都要。

### 3.3 成條 pathway 回顧（報告骨幹）

| 階段 | 由 | 到 | 主要貢獻 |
|---|---|---|---|
| Baseline | 0.408（平均價） | 0.155（Ridge） | 有 model |
| Feature engineering | 0.155 | 0.128 | **log 面積（12.4%）**、拎走 outlier、log target |
| Model 選擇 | 0.128（Ridge） | 0.123（XGBoost） | Boosting 處理非線性同交互作用 |
| Blend | 0.123 | **0.119** | 唔同家族嘅 model 錯誤互補 |

**由 Ridge baseline 到 Step 12 blend，CV 誤差總共跌咗 23.4%（0.155 → 0.119）。** Kaggle 分數同 CV 唔完全一致，原因同跟住點處理，見 Part 6。

---

## 4. 交 Kaggle 同記錄

`v2/results/kaggle_log.csv` 已經預先填好 6 個 submission 同 CV 分數，你只需要填 `kaggle_score`、`submitted_on`（日期）同 `notes`：

| submission | step | cv_rmse | kaggle_score |
|---|---|---|---|
| s1_ridge_raw | 1 | 0.15534 | ？ |
| s2_new_features | 4 | 0.15121 | ？ |
| s3_log_features | 5 | 0.13239 | ？ |
| s4_no_outliers | 6b | 0.12766 | ？ |
| s5_catboost | 10 | 0.12320 | ？ |
| s6_blend | 12 | 0.11906 | ？ |

**報告「best improvement」嗰題點揀：**
- 睇 Kaggle 分數，搵相鄰兩次 submission 之間進步最大嘅一次。
- 按 CV 預計係 **S2 → S3**：只改咗一樣嘢（log 面積），進步 12.4%，最容易清楚解釋。
- 如果 Kaggle 結果唔同，例如 S5 → S6 進步更大，就用 Kaggle 嘅結果，並且討論點解 CV 同 leaderboard 唔完全一致。CV 有選擇偏差（所有決定都係用佢揀嘅），leaderboard 就係另一批數據，有自己嘅隨機誤差。

---

## 5. 講師可能會問

**Q：咩係 blending？點解有用？**
A：將幾個 model 嘅預測平均。只要佢哋嘅錯誤唔完全相關，有啲估高、有啲估低，平均就可以互相抵消，所以比最好嘅單一 model 更準。我哋用 OOF residual 嘅 correlation 去量度呢種互補程度。

**Q：點解用等權重，唔用優化過嘅權重？**
A：權重只能夠喺同一批 OOF 預測上面揀，再用同一批數據報分數，就會 overfit 嗰 1460 間屋嘅雜訊，分數虛高。等權重唔使睇分數都可以決定，冇呢個偏差。

**Q：點揀 blend 入面有邊幾個 model？**
A：喺睇 blend 分數之前先定規則：每個家族揀最好嗰個，兩個 boosting 因為處理類別方法唔同所以都要。結果係 Lasso、XGBoost、CatBoost、SVR。坦白講，規則係睇過每個單一 model 嘅分數先定（例如知道 RF 弱），但冇睇過任何 blend 嘅分數。

**Q：點解加多啲 model 反而會差？**
A：Ridge、Lasso、ElasticNet 錯得幾乎一樣（correlation 0.99），三個一齊放入去，等於線性 model 有三票，平均會偏向佢哋嘅錯誤。多樣性比數量重要。

**Q：Blend 嘅 CV 分數可信嗎？**
A：大致可信，因為用嘅全部係 OOF 預測。但整個項目所有決定（特徵、參數、blend 成員）都係用同一個 CV 做嘅，所以分數會稍為樂觀。Kaggle leaderboard 係獨立嘅檢驗。

**Q：成個項目最重要嘅發現係咩？**
A：(1) 對偏斜面積取 log 係最有效嘅 feature engineering，因為佢令樓價同面積嘅比例關係變成直線；(2) boosting 比 bagging 好好多，因為呢個問題嘅主要難處係 bias；(3) 將錯誤互補嘅 model 平均，可以免費再進步 3%。

---

## 課堂連結

- Ensemble methods：combining classifiers / regressors、diversity
- Model evaluation：overfitting to validation data、selection bias
- Data visualization：heatmap、bar chart
