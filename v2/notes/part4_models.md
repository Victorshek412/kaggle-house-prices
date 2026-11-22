# Part 4 學習筆記：比較 7 個 model

> **前置知識（唔識就先睇 Part 0）：** dict → §1.2；function → §1.3；for 迴圈（包括兩層）→ §1.4；fit / predict → §4.2；coefficient → §4.3；overfitting / underfitting → §4.4；hyperparameter 同 tuning → §4.5；Pipeline → §4.6。
> 另外會用到 Part 2 嘅 `cv_evaluate`、`record`、`make_linear_preprocessor`，同 Part 3 嘅 `X_final`、`X4`、`outlier`。

---

## 0. 成個 Part 嘅思路同結果總覽

### 0.1 思路

Part 3 已經將**特徵定好**。Part 4 反過嚟做：**特徵唔郁，只換 model**。所有 model 都用：
- 同一個 5-fold 切法；
- 預測 log 價錢；
- 訓練時拎走 2 個 outlier。

所以分數嘅分別，**只會嚟自 model 本身**。呢個就係 Part 3 講過嘅「控制變因」原則。

### 0.2 結果（由好到差）

| 排名 | Model | 家族 | CV RMSE | 跑幾耐 |
|---|---|---|---|---|
| 1 | **XGBoost** | 樹（boosting） | **0.1231** | 4 秒 |
| 2 | **CatBoost** | 樹（boosting） | **0.1232** | 80 秒 |
| 3 | SVR | Kernel | 0.1235 | 5 秒 |
| 4 | Lasso | 線性 | 0.1255 | 2 秒 |
| 5 | ElasticNet | 線性 | 0.1257 | 1 秒 |
| 6 | Ridge | 線性 | 0.1277 | 0.4 秒 |
| 7 | Random Forest | 樹（bagging） | 0.1382 | 12 秒 |

**三個最重要嘅發現：**
1. **Boosting 最好**，XGBoost 同 CatBoost 幾乎打和。
2. **Random Forest 最差**，連最簡單嘅 Ridge 都輸。點解樹 model 入面會有一好一差，係呢個 Part 最值得理解嘅地方（§4.3）。
3. **頭 6 個 model 其實好接近**（0.123–0.128），所以 Part 5 將佢哋「夾埋」先有意思。

### 0.3 留意：成績表入面嘅「vs previous step」

`record` 每次都會同成績表上一行比較。Part 3 嘅 step 係一步疊一步，所以比較有意思；但 Part 4 每個 model 都係獨立嘅，「同上一個 model 比」就冇意思。例如 Step 8 嘅 +0.01254，只係話 Random Forest 差過 ElasticNet。

**Part 4 請睇最尾嗰張比較表同比較圖。**

---

## 1. 開頭嘅工具

### 1.1 逐行解釋

```python
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
```
- `time` 用嚟計每個 model 跑幾耐。
- 其餘係 5 個新 model。
- XGBoost 同 CatBoost 唔屬於 sklearn，係獨立嘅套件。不過佢哋都跟足 sklearn 嘅 `fit`/`predict` 規矩，所以可以直接放入 `cv_evaluate`。

```python
oof, test_preds, seconds = {}, {}, {}
```
開 3 個空 dict（→ Part 0 §1.2），key 係 model 名，例如 `"xgb"`：
- `oof`：每個 model 嘅 1460 個 OOF 預測（Part 5 要用嚟比較同 blend）；
- `test_preds`：每個 model 嘅 1459 個 test 預測；
- `seconds`：每個 model 跑咗幾多秒。

```python
def run_model(step, name, description, model, X_train, X_predict):
    started = time.time()
```
- 將「評分 → 存預測 → 計時 → 記錄」包成一個 function。7 個 model 都要做同一套嘢，寫一次就夠。
- `time.time()` 會回傳「而家係第幾秒」。開始記低一次，完咗再記一次，兩個相減就係用咗幾耐。

```python
    score, oof[name], test_preds[name] = cv_evaluate(model, X_train, y, X_predict,
                                                      target="log", drop_outliers=True)
```
`cv_evaluate` 回傳 3 樣嘢，**直接放入 3 個唔同位置**：分數放入 `score`，OOF 預測放入 `oof` 嘅 `name` 格，test 預測放入 `test_preds` 嘅 `name` 格。

```python
    seconds[name] = time.time() - started
    record(step, description, score)
    print(f"   ({seconds[name]:.0f} s)")
```
記時間，記成績，印出用咗幾多秒。`:.0f` 代表唔要小數。

```python
def linear_pipeline(model):
    return Pipeline([("prep", make_linear_preprocessor()), ("model", model)])
```
- 「預處理 + model」嘅 pipeline 會用好多次，所以寫個短 function。
- 叫 linear 係因為用 `make_linear_preprocessor`（填缺失 + 標準化 + one-hot），但 Random Forest、XGBoost、SVR 都會借用佢。

```python
run_model("6b", "ridge", "Ridge (reference, re-run)", linear_pipeline(Ridge(alpha=best_alpha)),
          X_final, X_test_final)
results = results.iloc[:-1]
```
- 重新跑一次 Part 3 最好嗰個 Ridge，目的係將佢嘅預測存入 `oof["ridge"]`，Part 5 要用。
- 但呢個結果同 6b 一模一樣，唔想成績表有重複，所以 `results.iloc[:-1]` 拎走最後一行（`[:-1]` 即係「由頭到倒數第二個」）。

---

## 2. Step 7：Lasso 同 ElasticNet

### 2.1 概念：三種 penalty

Part 2 講過 Ridge 嘅 penalty 係「係數平方」。另外仲有兩種：

```
Ridge:       盡量細：誤差² 總和  +  alpha × Σ (係數)²
Lasso:       盡量細：誤差² 總和  +  alpha × Σ |係數|
ElasticNet:  兩樣都有，用 l1_ratio 決定比例
```

**一個字之差（平方 vs 絕對值），行為好唔同：**

| | Ridge（平方） | Lasso（絕對值） |
|---|---|---|
| 對細係數 | 0.1² = 0.01，罰得好輕 | \|0.1\| = 0.1，罰得相對重 |
| 結果 | 所有係數縮細，但**唔會變成 0** | 冇用嘅係數會被壓到**啱啱好係 0** |
| 等於 | 「每個特徵都用少少」 | 「自動揀特徵」（feature selection） |

**直覺：** 用平方嚟罰，係數越接近 0，懲罰就跌得越快（0.01 → 0.0001），所以 Ridge 冇動力將係數壓到真正嘅 0。用絕對值，就算係數已經好細，懲罰都係同一個比例咁減。Lasso 會發現「完全唔用呢個特徵」最划算，所以直接將佢設做 0。

**ElasticNet：** 兩種 penalty 混合。`l1_ratio=0.5` 即係一半一半。當有幾個高度相關嘅特徵（例如 GarageCars 同 GarageArea），Lasso 通常會隨機揀一個、放棄其他；ElasticNet 傾向保留成組。

**點解 alpha 咁細（0.0005，而 Ridge 係 10）？** sklearn 寫 Lasso 公式嘅方式唔同：佢將誤差部份除咗數據行數（1458），所以同一個 alpha 數值嘅意義完全唔同，**兩者唔可以直接比較**。

### 2.2 逐行解釋

```python
for alpha in [0.0002, 0.0005, 0.001]:
    score, _, _ = cv_evaluate(linear_pipeline(Lasso(alpha=alpha, max_iter=50000)),
                              X_final, y, X_test_final, drop_outliers=True)
    print(f"Lasso alpha={alpha}: {score:.5f}")
```
- 同 Part 3 調 Ridge alpha 一樣嘅方法：試 3 個值，印出分數。
- `max_iter=50000`：Lasso 唔係一步計出答案，而係一步步逼近答案。alpha 細嘅時候要行好多步先收斂，預設嘅 1000 步會唔夠，仲會出警告。
- 呢度冇用 `record`：只係「試」，唔係正式 step。
- 結果 0.0005 最好，而且喺中間，唔係邊緣。

```python
run_model("7", "lasso", "Lasso (alpha=0.0005)",
          linear_pipeline(Lasso(alpha=0.0005, max_iter=50000)), X_final, X_test_final)
run_model("7b", "elasticnet", "ElasticNet (alpha=0.001, l1_ratio=0.5)",
          linear_pipeline(ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=50000)),
          X_final, X_test_final)
```
用揀好嘅設定正式評分同記錄。ElasticNet 嘅 alpha 用 0.001，因為佢只有一半係 Lasso penalty，所以大約要兩倍先有相若嘅力度。

```python
lasso_fit = linear_pipeline(Lasso(alpha=0.0005, max_iter=50000)).fit(X_final[~outlier], np.log(y[~outlier]))
coefficients = lasso_fit.named_steps["model"].coef_
print(f"Lasso kept {np.sum(coefficients != 0)} of {len(coefficients)} columns (the rest are exactly 0)")
```
呢 3 行唔係評分，係**睇入 model 入面**，驗證 Lasso 真係揀咗特徵：
- 用全部（除咗 outlier 嘅）訓練數據 fit 一次。`X_final[~outlier]` 用 boolean mask 揀行（→ Part 0 §2.3）。
- `.named_steps["model"]` 攞返 pipeline 入面叫 `"model"` 嗰一步，即係 Lasso 本身。
- `.coef_` 係佢學到嘅所有係數（→ Part 0 §4.3）。sklearn 慣例係：fit 完先有嘅屬性，名都以 `_` 結尾。
- `coefficients != 0` 得出一串 True/False；`np.sum` 數有幾多個 True。

### 2.3 點睇結果

- **Lasso 0.1255，好過 Ridge 0.1277。** 324 欄入面，Lasso 只用咗 107 欄，其餘 217 欄（大部份係罕見類別嘅 one-hot 欄）被設做 0。呢啲欄幾乎全部係雜訊；Ridge 仲會俾佢哋少少權重，Lasso 就直接丟棄，所以 overfit 少啲。
- **ElasticNet 0.1257，同 Lasso 差唔多。** 呢份數據入面，「保留成組相關特徵」冇帶來額外好處。
- **報告可以講：** 「Lasso 自動將 324 個特徵減到 107 個，證明 one-hot 之後有大量冗餘特徵。」

---

## 3. 樹 model 嘅基本：決策樹（decision tree）

Step 8–10 都係由決策樹砌成，所以先講清楚一棵樹係點運作。

### 3.1 一棵樹點樣估價

```
                    OverallQual ≥ 8？
                   /                \
                 否                   是
          TotalSF > 2000？        TotalSF > 3000？
           /        \               /        \
      $130k      $180k          $250k      $380k
```

- 樹會問一連串「是 / 否」問題，每問一題就將屋分成兩堆。
- 最底嗰層叫**葉（leaf）**，估價就係**嗰堆訓練屋嘅平均價**。
- **點樣揀問題？** 每一步試晒所有欄、所有切點，揀「分完之後兩堆屋各自嘅價錢最一致」嗰個。

### 3.2 樹嘅特性（同線性 model 好唔同）

| 特性 | 解釋 | 後果 |
|---|---|---|
| **唔理尺度** | 「TotalSF > 2000？」同「log(TotalSF) > 7.6？」分出嚟嘅兩堆屋一模一樣 | 標準化、取 log 對樹冇影響 |
| **自動搵交互作用** | 「大**而且**高質素」係樹嘅自然結構 | 唔使人手整「質素 × 面積」呢類特徵 |
| **估價係階梯形** | 每片葉只有一個值 | 估唔到一條平滑直線，只能用好多級階梯去逼近 |
| **唔識外推** | 葉嘅值係訓練屋嘅平均 | 永遠唔會估出比訓練數據最貴嘅屋更貴嘅價錢 |
| **一棵深樹會死記** | 一直分落去，每片葉得一兩間屋 | 嚴重 overfit |

最後一點係關鍵：**單一棵樹好差**。所以實際上都會用好多棵樹。點樣「用好多棵」有兩種方法：bagging 同 boosting。

---

## 4. Step 8：Random Forest（bagging）

### 4.1 概念

**Bagging = Bootstrap aggregating**

1. **Bootstrap：** 由 1458 間屋入面「有放返」咁抽 1458 次。有啲屋會被抽中幾次，有啲一次都冇（大約 37% 唔會被抽中）。每棵樹都用自己嗰份隨機樣本去訓練。
2. **隨機揀欄：** 每次分叉只可以喺隨機揀出嘅部份欄入面揀（`max_features=0.3`，即係 30%）。咁樣每棵樹都唔同。
3. **Aggregating：** 300 棵樹各自估價，然後取**平均**。

**點解平均會好啲？** 每棵樹都 overfit，但每棵 overfit 嘅方向唔同（因為數據同欄都係隨機嘅），平均之後，隨機錯誤就互相抵消。呢個就係**減低 variance**。

類比：問 300 個人估一隻牛有幾重，每個人都可能估錯好多，但 300 個人嘅平均通常好準，前提係佢哋係**獨立**咁估。

### 4.2 逐行解釋

```python
run_model("8", "rf", "Random Forest (300 trees)",
          linear_pipeline(RandomForestRegressor(n_estimators=300, max_features=0.3,
                                                min_samples_leaf=2, n_jobs=-1, random_state=42)),
          X_final, X_test_final)
```
- `n_estimators=300`：300 棵樹。越多越穩定，但越慢；去到幾百棵之後，再加都冇乜分別。
- `max_features=0.3`：每次分叉只考慮 30% 嘅欄，令樹與樹之間更唔同。
- `min_samples_leaf=2`：每片葉最少要有 2 間屋，防止樹分得太盡（死記單一間屋）。
- `n_jobs=-1`：用晒電腦所有 CPU 核心，平行咁種樹，快啲。
- `random_state=42`：固定隨機性，每次跑結果都一樣。
- 用 `linear_pipeline`：樹本身唔需要標準化，但需要 one-hot（sklearn 嘅樹唔識直接處理文字），同埋需要填缺失。標準化對樹冇害，所以直接借用。

### 4.3 點睇結果：0.1382，全場最差，連 Ridge 都輸

**點解？** 呢個係成個 Part 最值得寫入報告嘅分析：

1. **數據本身好「線性」。** Part 3 之後，log 價錢同 log 面積、質素等等係接近直線嘅關係，呢個正正係線性 model 嘅強項。樹只可以用階梯逼近直線，必然有誤差（§3.2「階梯形」）。
2. **Bagging 只減 variance，唔減 bias。** 每棵樹都有同一個系統性弱點：階梯形、唔識外推。300 棵樹平均，個弱點仍然喺度。平均只可以抵消**隨機**錯誤，抵消唔到**大家都犯嘅**錯誤。
3. **貴樓估唔到。** 最貴嘅屋喺葉入面會同稍平嘅屋平均，所以佢哋嘅價錢一定被低估，兩頭都會被「壓扁」。

> **Bias vs variance 一句講晒：** variance 係「每次訓練結果都唔同」嘅錯誤，平均可以抵消；bias 係「model 本身做唔到」嘅錯誤，平均冇用。Random Forest 對付 variance，但呢個問題嘅主要難處係 bias。

---

## 5. Step 9：XGBoost（boosting）

### 5.1 概念：逐棵樹去改錯

**Boosting 嘅做法同 bagging 完全相反：**

| | Bagging（Random Forest） | Boosting（XGBoost、CatBoost） |
|---|---|---|
| 樹點樣種 | 同時、各自獨立 | **一棵接一棵** |
| 每棵樹學咩 | 都係學估價 | 學**前面啲樹仲錯緊嘅部份** |
| 每棵樹嘅大小 | 深、大 | 淺、細（depth 3） |
| 主要減咩 | variance | **bias** |
| 點樣組合 | 平均 | 加埋 |

**一步步嘅例子**（一間真實 log 價錢係 12.0 嘅屋）：

```
開始：     估所有屋嘅平均        → 估 11.8，錯 +0.20
第 1 棵：  學錯誤，佢估 +0.18，只加 3%（learning rate）→ 11.8 + 0.03×0.18 = 11.805
第 2 棵：  學新嘅錯誤（+0.195）……   → 11.811
……
第 1500 棵：                         → 好接近 12.0
```

**Learning rate（學習率）0.03：** 每棵樹只係加佢估計嘅 3%，咁樣每一步都好細，由好多棵細樹慢慢一齊修正。點解唔一次過全加？因為每棵樹都有雜訊，一次加晒好容易 overfit；細步行可以令雜訊互相抵消。代價係要好多棵樹，所以係 1500 棵。

**點解 boosting 喺呢度贏？** 佢逐步修正系統性嘅錯誤（bias），包括樹天生嘅「階梯」問題：好多棵細樹加埋，可以砌出接近平滑嘅曲線。

### 5.2 逐行解釋

```python
run_model("9", "xgb", "XGBoost (1500 trees, depth 3, lr 0.03)",
          linear_pipeline(XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=3,
                                       subsample=0.8, colsample_bytree=0.5, min_child_weight=2,
                                       n_jobs=-1, random_state=42)),
          X_final, X_test_final)
```
- `n_estimators=1500`：種 1500 棵樹。
- `learning_rate=0.03`：每棵樹只加 3%（見上面）。learning rate 同樹數要一齊諗：learning rate 細，就要多啲樹。
- `max_depth=3`：每棵樹最多問 3 層問題（最多 8 片葉）。boosting 要用**淺**樹，每棵只負責修正少少，唔需要（亦唔應該）一棵樹就學晒。
- `subsample=0.8`：每棵樹隨機用 80% 嘅屋，引入少少 bagging 嘅隨機性，防止 overfit。
- `colsample_bytree=0.5`：每棵樹隨機用 50% 嘅欄，原因同上。
- `min_child_weight=2`：每片葉最少要有一定數量嘅數據，防止為一兩間屋特登開一片葉。
- 呢啲數值係常見嘅合理設定，**冇做大範圍調參**，因為時間有限，而且調得太盡會令 CV 太樂觀。

### 5.3 點睇結果：0.1231，全場第一

**同 Random Forest 用嘅係同一啲樹、同一啲特徵，分別只係「點樣組合」，分數就由 0.138 變成 0.123。** 報告值得寫：「model 嘅組合方式，可以比 model 嘅基本單位更重要。」

而且只用咗 4 秒，比 Random Forest 仲快。原因係每棵樹好淺，而 XGBoost 嘅實作本身好高效。

---

## 6. Step 10：CatBoost（直接處理類別嘅 boosting）

### 6.1 概念

CatBoost 都係 gradient boosting，原理同 XGBoost 一樣，但有兩個特別之處：

**1. 直接處理文字欄（唔使 one-hot）**

one-hot 嘅問題：Neighborhood 有 25 個地區，就會變成 25 欄 0/1。罕見嘅地區得幾間屋，好難學。

CatBoost 嘅做法係將每個類別換成「**呢個類別嘅屋，平均價錢係幾多**」，叫 target statistics。例如 NoRidge 地區平均 log 價錢係 12.6，就用 12.6 代替 "NoRidge"。

**但咁樣會 leak！** 計平均時，如果包括咗呢間屋自己嘅價錢，就等於偷睇答案。CatBoost 嘅解決方法叫 **ordered target statistics**：將屋隨機排好次序，每間屋只用「排喺佢前面」嘅屋計平均。咁樣每間屋都唔會用到自己嘅答案。仲會加 smoothing：屋好少嘅類別，會向整體平均靠攏。

**2. 對稱樹（symmetric / oblivious trees）**

同一層嘅所有分叉都問**同一條問題**。例如第一層全部問「OverallQual ≥ 7？」，第二層全部問「TotalSF > 2000？」。呢種樹比較簡單、限制多，所以喺細數據（1460 行）上唔容易 overfit。

### 6.2 逐行解釋

```python
categorical_columns = X4.select_dtypes(exclude="number").columns.tolist()
```
- 用 X4，即係 Part 3 Step 4 嘅特徵：已經填好有意思嘅缺失、加咗新特徵，但**未取 log**。樹唔使 log（§3.2），用原始尺度反而易解釋。
- `select_dtypes(exclude="number")` 揀出所有非數字欄，即係文字欄（44 欄，包括已經轉做文字嘅 MSSubClass）。
- `.columns.tolist()` 將欄名變成普通 list。

```python
X_cat = X4.copy()
X_cat_test = X4_test.copy()
X_cat[categorical_columns] = X_cat[categorical_columns].fillna("Missing")
X_cat_test[categorical_columns] = X_cat_test[categorical_columns].fillna("Missing")
```
- 先複製一份，唔郁 X4（→ Part 0 §3.4）。
- CatBoost 要求類別欄冇 NaN，所以填 `"Missing"`。數字欄嘅 NaN 就唔使填，CatBoost 自己識處理。
- 呢度喺 CV 外面填係安全嘅：填一個固定字串，同其他屋無關，係逐行操作（Part 3 §1）。

```python
catboost = CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                             cat_features=tuple(categorical_columns), random_seed=42,
                             verbose=0, allow_writing_files=False)
```
- `iterations=1500`：1500 棵樹，即係 XGBoost 嘅 `n_estimators`。
- `depth=6`：對稱樹限制多，所以可以深啲。
- `l2_leaf_reg=3`：葉值嘅 L2 penalty，即係同 Ridge 一樣嘅「平方懲罰」概念，用喺葉上面。
- `cat_features=tuple(...)`：話俾 CatBoost 知邊啲係類別欄。**點解要用 tuple，唔用 list？** 呢個係我實際撞到嘅問題：sklearn 嘅 `clone`（`cv_evaluate` 每個 fold 都會用）複製唔到 cat_features 係 list 嘅 CatBoost，會報錯。改用 tuple 就得。呢類「工具之間嘅相容性問題」係真實世界常見嘅困難，可以寫入個人答案。
- `verbose=0`：唔好每棵樹都印進度。
- `allow_writing_files=False`：唔好喺 folder 度寫訓練記錄檔。
- **點解唔用 Pipeline？** 因為唔需要 one-hot 同標準化：CatBoost 自己處理類別，樹唔理尺度。

```python
run_model("10", "catboost", ..., catboost, X_cat, X_cat_test)
save_submission("s5_catboost", test_preds["catboost"])
```
評分同記錄，再出 submission **S5**。

### 6.3 點睇結果：0.1232，同 XGBoost 幾乎打和

- 兩者都係 boosting，所以表現相近合理。
- CatBoost 冇用 log 特徵、冇用 one-hot，但結果一樣好。即係話，**樹 model 唔需要 Part 3 嘅 log 處理**，同 §3.2 講嘅一致。
- 代價係慢好多：80 秒，XGBoost 只要 4 秒。原因係 ordered target statistics 要額外計算。
- 點解仲要兩個都留？因為佢哋處理類別嘅方法唔同（one-hot vs target statistics），犯嘅錯可能唔同。Part 5 會驗證呢點。

---

## 7. Step 11：SVR（Support Vector Regression）

### 7.1 概念

**核心：用「相似度」估價。** 要估一間新屋，SVR 會睇佢同每一間訓練屋有幾「似」，似嘅屋影響大，唔似嘅屋影響細。

**「似」點樣量度？** 用 RBF kernel：

```
相似度 = exp( −gamma × 兩間屋特徵之間嘅距離² )
```

- 距離 = 0（完全一樣）→ 相似度 1。
- 距離越遠 → 相似度越接近 0。
- **gamma** 決定「幾遠就當唔似」：
  - gamma 大：只有好近嘅屋先算似，曲線好彎，容易 overfit；
  - gamma 細：好遠都算似，曲線好平，容易 underfit。

**點解一定要標準化？** 距離係所有特徵差距嘅總和。唔標準化嘅話，LotArea（幾萬）嘅差距會蓋過晒 OverallQual（1–10）嘅差距，等於只睇地皮面積。我哋嘅 pipeline 有 StandardScaler，所以冇問題。

**epsilon 同 C：**
- **epsilon（ε）**：誤差細過 ε（0.01，即係大約 1%）就當冇錯，唔罰。呢個「容忍管」令 model 唔使遷就每一個細誤差。
- **C**：超出容忍管嘅誤差罰幾重。
  - C 大：好努力去 fit，容易 overfit；
  - C 細：容許多啲誤差，model 簡單啲。
  - 留意：C 嘅方向同 Ridge 嘅 alpha **相反**。C 越大，正則化越弱。

### 7.2 逐行解釋

```python
svr_scores = {}
for C in [1, 3, 10, 30]:
    for gamma in [0.0001, 0.0003, 0.001, 0.003]:
        svr_scores[(C, gamma)], _, _ = cv_evaluate(
            linear_pipeline(SVR(C=C, epsilon=0.01, gamma=gamma)),
            X_final, y, X_test_final, drop_outliers=True)
        print(f"SVR C={C}, gamma={gamma}: {svr_scores[(C, gamma)]:.5f}")
```
- **兩層 for 迴圈**：外層逐個 C，內層逐個 gamma，總共 4 × 4 = 16 個組合，呢種做法叫 **grid search（格網搜尋）**。
- `svr_scores[(C, gamma)]`：dict 嘅 key 可以係一個 tuple（一對數），咁就記到「邊對參數 → 幾多分」。
- 兩個參數都係每次大約 ×3，同 Part 3 Ridge alpha 嘅道理一樣（對數尺度）。

```python
best_C, best_gamma = min(svr_scores, key=svr_scores.get)
```
- 搵分數最細嗰個 key，即係 (C, gamma) 一對數（同 Part 3 嘅 `best_alpha` 一樣寫法）。
- 因為 key 係 tuple，可以直接拆開放入兩個名（unpacking）。

```python
run_model("11", "svr", f"SVR (C={best_C}, gamma={best_gamma})",
          linear_pipeline(SVR(C=best_C, epsilon=0.01, gamma=best_gamma)), X_final, X_test_final)
```
用最好嘅參數正式記錄。

### 7.3 調參過程（呢個過程本身就係 pathway）

| 次數 | C 試咗 | gamma 試咗 | 最好嘅組合 | 問題 |
|---|---|---|---|---|
| 第 1 次 | 10, 30 | 0.0003, 0.001 | C=10, γ=0.0003 | 兩個都喺邊緣 |
| 第 2 次 | 3, 10, 30 | 0.0001–0.001 | C=3, γ=0.001 | 又喺邊緣 |
| 第 3 次 | 1, 3, 10, 30 | 0.0001–0.003 | **C=1, γ=0.003** | 喺角落，要再驗證 |
| 驗證 | 0.3, 1 | 0.003–0.01 | 外面全部更差 | ✓ 真正嘅最低點 |

- **點解會一路「走位」？** C 同 gamma 會互相抵銷：「C 細 + gamma 大」同「C 大 + gamma 細」嘅效果相似。所以好嘅組合係沿住一條斜線分佈，唔係集中喺一點。
- **教訓：** 調參一定要檢查最好嘅值係咪喺邊緣；如果係，要擴大範圍。
- **代價：** 試得越多組合，越有機會揀中「啱啱好喺呢 5 個 fold 表現好」嘅值，CV 分數就越樂觀。所以我哋只試咗 16 + 5 個組合，冇無限咁試落去。

### 7.4 點睇結果：0.1235，第三

- 比所有線性 model 都好，因為 RBF kernel 可以捕捉彎曲嘅關係同交互作用。
- 但好**敏感**：最好同最差嘅組合由 0.123 去到 0.145。所以 SVR 一定要調參，唔可以用預設值。
- SVR 用嘅係同線性 model 一樣嘅特徵（X_final），但「似鄰居」嘅思路完全唔同，Part 5 blend 時可能會帶嚟幫助。

---

## 8. 比較表同比較圖

```python
names = ["ridge", "lasso", "elasticnet", "rf", "xgb", "catboost", "svr"]
y_log = np.log(y.values)
comparison = pd.DataFrame({
    "cv_rmse": [np.sqrt(np.mean((y_log - oof[name]) ** 2)) for name in names],
    "seconds": [seconds[name] for name in names],
}, index=names).sort_values("cv_rmse")
```
- `[... for name in names]` 叫 **list comprehension**：即係一個縮短咗嘅 for 迴圈，逐個 name 計一個數，結果組成一個 list。等於：
  ```python
  scores = []
  for name in names:
      scores.append(np.sqrt(np.mean((y_log - oof[name]) ** 2)))
  ```
- 每個 model 嘅 RMSE 都由佢嘅 OOF 預測重新計，結果同 `record` 記低嘅一樣，只係方便放入同一張表。
- `index=names`：用 model 名做行名。
- `.sort_values("cv_rmse")`：由好到差排。

```python
ax.barh(comparison.index[::-1], comparison.cv_rmse[::-1], color="steelblue")
ax.set(..., xlim=(comparison.cv_rmse.min() * 0.95, comparison.cv_rmse.max() * 1.02))
```
- `[::-1]` 倒轉次序，令最好嘅 model 喺圖嘅最上面（barh 係由下而上畫）。
- `xlim` 設定 x 軸範圍：唔由 0 開始，而係由最細值嘅 95% 開始，咁差異先睇得清楚。**報告用呢類圖要講清楚 x 軸唔係由 0 開始**，否則會誇大差異。

---

## 9. 交 Kaggle

| Submission | 檔案 | Model | CV |
|---|---|---|---|
| S5 | `s5_catboost.csv` | CatBoost | 0.1232 |

只交 CatBoost，係因為佢係第一個「樹 model」嘅代表，可以同 S4（Ridge）比較線性 vs 樹喺 leaderboard 上嘅分別。XGBoost CV 稍好，但兩者差 0.0001，冇統計上嘅分別；而且 Kaggle 每日 submission 次數有限。最終嘅 S6 blend 會將所有好嘅 model 都用埋。

---

## 10. 講師可能會問

**Q：Bagging 同 boosting 有咩分別？**
A：Bagging（例如 Random Forest）同時、獨立咁種好多棵深樹，然後取平均，主要減 variance。Boosting（例如 XGBoost、CatBoost）一棵接一棵咁種淺樹，每棵學前面仲錯緊嘅部份，然後加埋，主要減 bias。

**Q：點解 Random Forest 咁差，仲差過 Ridge？**
A：經過 Part 3 嘅 log 轉換，log 價錢同特徵之間係接近線性嘅關係。樹只能用階梯形去逼近直線，而且唔識外推。Bagging 平均只能抵消隨機錯誤（variance），抵消唔到所有樹都有嘅系統性錯誤（bias）。Boosting 逐步修正 bias，所以同樣係樹，XGBoost 就好好多。

**Q：Lasso 同 Ridge 有咩分別？點解 Lasso 好啲？**
A：Ridge 罰係數平方，會縮細所有係數但唔會變 0；Lasso 罰絕對值，會將冇用嘅係數壓到 0，等於自動揀特徵。324 個特徵入面，Lasso 只保留咗 107 個，其餘大多係罕見類別嘅 one-hot 欄。去除呢啲雜訊之後，overfit 少咗。

**Q：點解樹 model 唔需要標準化同取 log？**
A：樹用「特徵 > 某個值？」嚟分數據。標準化同 log 都唔會改變數值嘅大細次序，所以分出嚟嘅結果一樣。

**Q：點解 SVR 一定要標準化？**
A：RBF kernel 用特徵之間嘅距離量度相似度。唔標準化嘅話，數值大嘅特徵（例如 LotArea）會主導成個距離。

**Q：CatBoost 點樣處理類別特徵？會唔會 leak？**
A：佢用 target statistics，即係將類別換成該類別嘅平均價錢。為咗避免 leak，佢用 ordered 方法：每間屋只用排喺佢前面嘅屋計平均，唔會用到自己嘅價錢。

**Q：你哋點樣調參？**
A：用同一個 5-fold CV 做 grid search，參數按對數尺度揀。如果最好嘅值喺範圍邊緣，就擴大範圍。SVR 擴大咗兩次，最後喺範圍外面再驗證，確認係真正嘅最低點。我哋刻意限制試嘅組合數量，避免 CV 分數過度樂觀。

**Q：最好嘅單一 model 係邊個？**
A：XGBoost（0.1231）同 CatBoost（0.1232），兩者幾乎一樣。

**Q：（個人問題）遇到咩困難？**
A：可以講兩樣：(1) CatBoost 同 sklearn 嘅 clone 唔相容，報錯之後要讀錯誤訊息，先發現原因係 cat_features 嘅資料類型；(2) SVR 調參時，最好嘅值一直喺格網邊緣，要明白參數之間會互相抵銷，先知道點樣有系統咁擴大搜尋。

---

## 課堂連結

- Classification / regression trees：decision tree、splitting criteria
- Ensemble methods：bagging、random forest、boosting
- Support vector machines：kernel、margin
- Regularization：L1（Lasso）vs L2（Ridge）
- Model selection：grid search、bias-variance trade-off
