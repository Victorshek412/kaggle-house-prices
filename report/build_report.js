// Builds report/House_Prices_Report.docx: single column with a table of contents,
// Times New Roman 12. Run from the repo root after `python report/make_figures.py`:
//     node report/build_report.js
// Word fills in the table of contents when it opens the file (it asks to update fields; say yes).
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, Header, Footer,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, PageNumber, TableOfContents, PageBreak,
} = require("docx");

const FONT = "Times New Roman";
const FIGS = path.join(__dirname, "figures");
const V2 = path.join(__dirname, "..", "v2");
const FIG_PX = 380;         // default figure width; the text width is about 650 px

// ---------- helpers ----------
const run = (text, o = {}) => new TextRun({ text, font: FONT, size: 24, ...o });

// body("text") or body(["plain ", {b: "bold"}, {i: "italic"}])
function body(parts, o = {}) {
  const list = Array.isArray(parts) ? parts : [parts];
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 100, line: 240 },
    ...o,
    children: list.map((p) => typeof p === "string" ? run(p)
      : p.b ? run(p.b, { bold: true }) : run(p.i, { italics: true })),
  });
}

// numbered list item with a hanging indent
const item = (n, text) => body(`${n}. ${text}`, { indent: { left: 360, hanging: 360 } });

const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });

function caption(kind, n, text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 60, after: 160 }, keepNext: kind === "Table",
    children: [run(`${kind} ${n}. `, { bold: true, size: 20 }), run(text, { italics: true, size: 20 })],
  });
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
}

function figure(file, n, text, widthPx = FIG_PX) {
  const { width, height } = pngSize(file);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 100, after: 0 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file),
        transformation: { width: widthPx, height: Math.round((widthPx * height) / width) } })],
    }),
    caption("Figure", n, text),
  ];
}

// booktabs-style table, caption above: rules above and below the header and at the bottom
function table(n, captionText, headers, rows, widths, align = null) {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const thick = { style: BorderStyle.SINGLE, size: 12, color: "000000" };
  const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
  const all = [headers, ...rows];
  const centred = (j) => align ? align[j] === "c" : /^[-+−$0-9.,% s()]+$/.test(all[1][j]);
  const make = (cells, i, last) => new TableRow({
    cantSplit: true,
    children: cells.map((text, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA },
      margins: { top: 30, bottom: 30, left: 80, right: 80 },
      borders: { left: none, right: none,
        top: i === 0 ? thick : i === 1 ? thin : none,
        bottom: last ? thick : none },
      children: [new Paragraph({
        keepNext: !last,
        alignment: centred(j) ? AlignmentType.CENTER : AlignmentType.LEFT,
        children: [run(text, { size: 21, bold: i === 0 })],
      })],
    })),
  });
  return [
    caption("Table", n, captionText),
    new Table({
      width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
      columnWidths: widths,
      alignment: AlignmentType.CENTER,
      borders: { insideHorizontal: none, insideVertical: none, top: none, bottom: none, left: none, right: none },
      rows: all.map((r, i) => make(r, i, i === all.length - 1)),
    }),
    new Paragraph({ spacing: { after: 120 }, children: [] }),
  ];
}

const rule = () => new Paragraph({
  spacing: { after: 100 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "000000", space: 1 } },
  children: [],
});

// ---------- numbers from the notebook's results table ----------
function readCsv(file) {
  return fs.readFileSync(file, "utf8").trim().split("\n").slice(1).map((line) => {
    const cells = [];
    let cur = "", quoted = false;
    for (const ch of line) {
      if (ch === '"') quoted = !quoted;
      else if (ch === "," && !quoted) { cells.push(cur); cur = ""; }
      else cur += ch;
    }
    return [...cells, cur];
  });
}
const cv = Object.fromEntries(readCsv(path.join(V2, "results", "cv_results.csv")).map(([s, d, v]) => [s, Number(v)]));
const f5 = (x) => x.toFixed(5);
const delta = (a, b) => { const d = cv[a] - cv[b]; return `${d < 0 ? "−" : "+"}${Math.abs(d).toFixed(5)}`; };

// ---------- code excerpts copied from the notebook source ----------
const SRC = path.join(V2, "src");
function sourceLines(file, startPattern, stopAfter = null) {
  const lines = fs.readFileSync(path.join(SRC, file), "utf8").split("\n");
  const i = lines.findIndex((l) => l.startsWith(startPattern));
  if (i < 0) throw new Error(`${startPattern} not found in ${file}`);
  if (stopAfter !== null) return lines.slice(i, i + stopAfter);
  const out = [lines[i]];
  for (let j = i + 1; j < lines.length; j++) {
    if (lines[j].trim() !== "" && !/^\s/.test(lines[j])) break;
    out.push(lines[j]);
  }
  while (out.length && out[out.length - 1].trim() === "") out.pop();
  return out;
}

function code(lines) {
  return lines.map((l, k) => new Paragraph({
    spacing: { after: 0, line: 220 },
    shading: { fill: "F2F2F2", type: "clear", color: "auto" },
    keepNext: k < lines.length - 1,
    children: [new TextRun({ text: l.length ? l : " ", font: "Courier New", size: 17 })],
  })).concat([new Paragraph({ spacing: { after: 120 }, children: [] })]);
}

const F = (name) => path.join(FIGS, name);
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

// ---------- page 1: title and summary; page 2: contents ----------
const FRONT = [
  rule(),
  new Paragraph({ spacing: { before: 120, after: 120 }, children: [run("House Price Prediction", { bold: true, size: 44 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [run("Kaggle competition: House Prices, Advanced Regression Techniques", { size: 26 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [run("[Team name]: [Name] ([Student ID]), [Name] ([Student ID])", { bold: true })] }),
  new Paragraph({ spacing: { after: 120 }, children: [run("[Course code and name]")] }),
  rule(),
  h1("Summary"),
  body("Our team entered the Kaggle competition House Prices: Advanced Regression Techniques. The task is to predict the sale prices of 1,459 houses in Ames, Iowa, from 79 features that describe each house. Our final submission scores 0.11624 on the leaderboard, which is rank 257."),
  body("Our first attempt was a pipeline generated with an AI coding assistant. It scored 0.11745, but we could not explain why each part was there. So we rebuilt the solution in 13 steps with two rules: each step changes one thing, and each step is scored with the same five-fold cross-validation (CV). That way we can explain every decision and say what it was worth. Table 1 lists the main results."),
  ...table(1, "Main results",
    ["Item", "Result"],
    [
      ["Final Kaggle score", "0.11624 (rank 257), submission S7"],
      ["First attempt", "0.11745 on Kaggle"],
      ["Best CV score", "0.1191, an equal-weight blend of Lasso, XGBoost, CatBoost and SVR"],
      ["Most useful feature engineering", "log of 20 skewed area features: CV 0.1512 to 0.1324 (−12%)"],
      ["Biggest improvement on Kaggle", "S1 to S2: 0.14188 to 0.13044 (−0.0114)"],
      ["Improvement we learned most from", "S6 to S7: 0.12037 to 0.11624, after an error analysis"],
    ],
    [3100, 6300], ["l", "l"]),
  pageBreak(),
  new Paragraph({ spacing: { after: 160 }, children: [run("Contents", { bold: true, size: 30 })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  pageBreak(),
];

// ---------- body ----------
const BODY = [
  // 1
  h1("1 Problem and Metric"),
  h2("1.1 The task and the data"),
  body("Kaggle gives us two files: train.csv with 1,460 houses and their sale prices, and test.csv with 1,459 houses without prices. We train on the first, predict a price for every house in the second, and upload the predictions as a CSV file with the columns Id and SalePrice. Kaggle compares them with the true prices, which only Kaggle has."),
  body("Each house has 79 features. 36 are numbers, such as the above-ground living area in square feet (GrLivArea), the year built and the number of bathrooms. The other 43 are text categories, such as the neighbourhood, the garage type and quality ratings from Po (poor) to Ex (excellent). The houses were sold between 2006 and 2010."),
  h2("1.2 How Kaggle scores us"),
  body("Kaggle scores a submission with the root mean squared error (RMSE) between the log of our predicted price and the log of the true price:"),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 40, after: 140 },
    children: [run("RMSE = √[ (1/n) × Σ (log predicted price − log true price)² ]", { italics: true })] }),
  body("For each house we take the difference between the two logs and square it, so errors above and below the true price both count. We average the squares over all n houses and take the square root. Lower is better."),
  body("Because the metric compares logs, it measures each error as a percentage of the price. Table 2 shows two predictions that are both 10% too high. One misses by $10,000 and the other by $50,000, yet both cost 0.095. Without the log, the few expensive houses would dominate the score. Our final score of 0.116 means a typical prediction is about 12% away from the true price."),
  ...table(2, "Two predictions that are both 10% too high get the same log error",
    ["House", "True price", "Prediction", "Error in dollars", "Log error"],
    [
      ["Cheaper house", "$100,000", "$110,000", "$10,000", "0.095"],
      ["Expensive house", "$500,000", "$550,000", "$50,000", "0.095"],
    ],
    [2000, 1800, 1800, 2000, 1500]),
  h2("1.3 How we worked"),
  body("Our first submission, on 9 November, came from a pipeline generated with an AI coding assistant (Codex). It ran 27 experiments, blended ElasticNet with CatBoost, and scored 0.1212 on CV and 0.11745 on Kaggle, but we could not explain its features or settings. We started again from an empty notebook and followed the order in Figure 1. We explored the data (Section 2), cleaned it (Section 3) and set up a fixed way to score every idea (Section 4). Then we improved the features one step at a time with the same simple model (Section 5), compared seven models and blended the best ones (Section 6). After submitting to Kaggle, we compared our CV scores with the leaderboard, found a blind spot and made one last change (Section 7)."),
  ...figure(F("f05_pathway.png"), 1, "Our pathway from the data to the final submission", 270),
  body("The notebook has six parts, one for each stage in Figure 1, and we wrote study notes for each part before moving on. Every step saves its CV score to a results table (Appendix B). We also wrote 17 automated tests, for example one that checks that the outlier step really changes the score. Our AI-use statement explains how we used AI tools."),

  // 2
  h1("2 Exploratory Data Analysis"),
  h2("2.1 Target variable"),
  body("SalePrice ranges from $34,900 to $755,000, and the middle half of the houses sold for $130,000 to $214,000. The distribution has a long right tail (Figure 2, left), with a skewness of 1.88; a symmetric shape would have 0. After a log transform the skewness is 0.12 and the shape is close to a bell curve (Figure 2, right). So we predict log(SalePrice). Kaggle scores on this scale anyway, and linear models work best when the target is roughly symmetric. We convert the predictions back to dollars with the exponential function at the very end."),
  ...figure(F("f01_target.png"), 2, "SalePrice before and after the log transform", 340),
  h2("2.2 Features most correlated with SalePrice"),
  body("Figure 3 shows the numeric features most correlated with SalePrice. Correlation measures how closely two numbers move together along a straight line, on a scale from −1 to 1. Overall quality (OverallQual, 0.79) and living area (GrLivArea, 0.71) lead, followed by garage and basement size, so size and quality explain much of the price. Several features measure almost the same thing, such as GarageCars and GarageArea, or TotalBsmtSF and 1stFlrSF. Overlapping features like these are one reason we use regression models with a penalty (Section 6.1)."),
  body("We did not drop any feature for a low correlation. Correlation only sees straight-line relationships between numbers, so it says nothing about the 43 text features, such as the neighbourhood, and it misses curved relationships."),
  ...figure(F("f04_correlation.png"), 3, "Numeric features with the highest correlation with SalePrice", 440),
  h2("2.3 Missing-value pattern"),
  body("Several features are mostly empty in the training set. PoolQC is missing (NA) for 1,453 of the 1,460 houses, MiscFeature for 1,406, Alley for 1,369 and Fence for 1,179. Others have a few gaps, such as LotFrontage with 259. The data description shows that NA means two different things, which Section 3 deals with. In addition, 15 features have gaps only in the test set, such as MSZoning and GarageCars, so our cleaning has to handle gaps it never sees in training."),
  h2("2.4 Outliers in GrLivArea"),
  body("Figure 4 plots living area against price. Most houses follow an upward trend, but two houses, Ids 524 and 1299, have more than 4,500 sq ft and sold for only $184,750 and $160,000. Both were Partial sales, which means the house was not finished when it was last assessed. The two other houses of that size sold for $745,000 and $755,000 and follow the trend, so we treat only the two cheap ones as outliers. They come back in Section 7."),
  ...figure(F("f03_outliers.png"), 4, "Living area against price; the circled houses are the outliers", 370),

  // 3
  h1("3 Data Cleaning"),
  h2("3.1 Values that are not really missing"),
  body("For many features, the data description says that NA means the house does not have the item: NA in PoolQC means no pool, and NA in GarageType means no garage. That is information, because a house without a garage is usually cheaper. Other features, such as LotFrontage, were simply not recorded. Table 3 gives examples of both."),
  ...table(3, "Two meanings of NA and how we handle them",
    ["Feature", "NA count (train)", "Meaning", "Handling"],
    [
      ["PoolQC", "1,453", "no pool", "label NoPool"],
      ["GarageType", "81", "no garage", "label NoGarage"],
      ["Fence", "1,179", "no fence", "label NoFence"],
      ["LotFrontage", "259", "not recorded", "median, inside the Pipeline"],
      ["MasVnrArea", "8", "not recorded", "median, inside the Pipeline"],
    ],
    [2000, 2100, 2000, 3300]),
  body("We replace NA with a label such as NoGarage only when another feature confirms it: GarageArea = 0 for the four garage text features, Fireplaces = 0 for the fireplace quality and PoolArea = 0 for the pool quality. Alley, Fence and MiscFeature have no size feature, so we follow the data description and label every NA. Table 4 shows the rule on three houses."),
  ...table(4, "The labelling rule on three houses (we removed Id 2's garage type to show the third case)",
    ["House", "GarageArea", "GarageType before", "GarageType after"],
    [
      ["Id 40", "0", "NA", "NoGarage"],
      ["Id 1", "548", "Attchd", "Attchd"],
      ["Id 2", "460", "NA", "NA (unknown)"],
    ],
    [1900, 1900, 2700, 2900]),
  body("By default, pandas also reads the text \"None\" as missing. MasVnrType = None is a real category (no masonry veneer), so with the default settings 872 houses looked unknown. We read both files with keep_default_na=False, which leaves only 8 truly unknown values."),
  h2("3.2 Truly missing values"),
  body("Some values were not recorded, such as LotFrontage (259 houses), MasVnrArea (8 houses) and the 15 test-only gaps. We fill missing numbers with the median of the training houses and missing text with the label \"Missing\". The median suits skewed features, because a few huge values, such as very large lots, pull the mean up but barely move the median. These fills are learned from the data, so we compute them inside a Pipeline that is refitted in every CV fold (Section 4.2)."),
  h2("3.3 Outliers"),
  body("A training house is an outlier if GrLivArea > 4,000 and SalePrice < $300,000, which matches only Ids 524 and 1299. In Step 6 we remove them from the training part of every CV fold. They still get a prediction and still count in the CV score, so the score stays honest. Section 7 explains why we later kept them and flagged them instead (Step 13)."),
  h2("3.4 Errors found by consistency checks"),
  body("Some features must agree with each other. A house with a garage area above 0 has a garage, and a garage cannot be built years after the house was sold. We checked pairs like these in both files (Table 5)."),
  ...table(5, "Consistency checks and what we did",
    ["Check", "Houses found", "What we did"],
    [
      ["Garage built more than a year after the sale", "test 2593 (built 2207)", "year treated as missing (Step 13)"],
      ["House built or remodelled after the sale", "train 524; test 2296, 2550", "negative ages set to 0"],
      ["Garage area above 0, garage details missing", "test 2127", "kept as missing, not NoGarage"],
      ["Pool area above 0, pool quality missing", "test 2421, 2504, 2600", "kept as missing, not NoPool"],
      ["Fireplaces above 0, fireplace quality missing", "none", "nothing needed"],
    ],
    [3800, 2800, 2800], ["l", "l", "l"]),
  body("The year 2207 is almost certainly a typo for 2007; left alone, it would look like a garage from the future. The sales before the build or remodel year are real. Houses 524 and 2550 are Partial sales, sold before they were finished. Our labelling rule from Section 3.1 already handles the garage and pool cases, because it only labels a gap when the size feature is 0."),

  // 4
  h1("4 Validation and Baseline"),
  h2("4.1 Five-fold cross-validation"),
  body("Kaggle only shows a test score after a submission, so we need our own way to score ideas. We use five-fold CV. The 1,460 training houses are split into five folds of 292, a model trained on the other 1,168 houses predicts one fold, and each fold takes a turn. Every house gets exactly one prediction from a model that never saw its price. These are the out-of-fold (OOF) predictions, and their RMSE is our CV score. For the test set we average the predictions of the five fold models."),
  body("We fix the folds with KFold(shuffle=True, random_state=42) and use the same folds for every step. With new folds at each step, a change in the score could come from the split instead of from our idea."),
  h2("4.2 Avoiding data leakage"),
  body("Data leakage happens when information from the houses being predicted gets into training, which makes the CV score look better than it is. Filling LotFrontage is an example: its median is 69 over all 1,460 houses but 70 over the training part of fold 1. Filling the gaps before splitting would let the validation houses help choose the fill value."),
  body("To prevent this, we put the median filling, the scaling and the one-hot encoding (one 0/1 column per text category) in one scikit-learn Pipeline. In each fold the Pipeline learns these values from the training houses only and applies them to the validation houses. The encoder ignores categories it has not seen, so a category that appears only in validation or test data cannot break it. Changes that use one house at a time, such as adding two of its areas, cannot leak, so we apply those before CV."),
  h2("4.3 Baselines"),
  body("Predicting the average price for every house gives a CV RMSE of 0.408 (Step 0). Ridge regression (Section 6.1) on the raw price, with only the Pipeline's basic cleaning, gives 0.155 (Step 1)."),

  // 5
  h1("5 Feature Engineering"),
  body("In Steps 2 to 6 we keep the same Ridge model (alpha = 10) and change one thing per step, so any change in the score comes from the features (Table 6)."),
  ...table(6, "Feature engineering steps (Ridge, five-fold CV RMSE)",
    ["Step", "Change", "CV RMSE", "Change in RMSE"],
    [
      ["1", "Raw price (baseline)", f5(cv["1"]), "-"],
      ["2", "Predict log(SalePrice)", f5(cv["2"]), delta("2", "1")],
      ["3", "Label NA by meaning; MSSubClass as text", f5(cv["3"]), delta("3", "2")],
      ["4", "TotalSF, TotalBathrooms, ages, TotalPorchSF", f5(cv["4"]), delta("4", "3")],
      ["5", "log1p of 20 skewed area features", f5(cv["5"]), delta("5", "4")],
      ["6", "Drop 2 outliers from training folds", f5(cv["6"]), delta("6", "5")],
    ],
    [900, 4900, 1700, 1900]),
  h2("5.1 Predicting the log price (Step 2)"),
  body("Step 2 trains Ridge on log(SalePrice) instead of the price (Section 2.1). The score improved by only 0.0033, less than we expected. Section 5.4 explains why: the log target pays off once the area features are on a log scale too."),
  h2("5.2 Encoding: missing labels and MSSubClass (Step 3)"),
  body("Step 3 applies the labelling rule from Section 3.1 and turns MSSubClass into text. MSSubClass stores the dwelling type as a number code, such as 20 for a one-storey house built after 1945 and 60 for a two-storey house built after 1945. As a number, a linear model would treat type 60 as three times type 20. As text, the one-hot encoder gives each code its own column, so each house type gets its own adjustment."),
  body("This step improved CV by only 0.0010, because the Pipeline already turned missing text into a \"Missing\" category that the model used much like our labels. We kept it because the labels are correct and CatBoost uses the text features directly."),
  h2("5.3 Combining features (Step 4)"),
  body("The data splits the size of a house over several features, while a buyer thinks about the total. In Step 4 we add five combined features:"),
  item(1, "TotalSF: living area plus basement area."),
  item(2, "TotalBathrooms: full bathrooms plus 0.5 for each half bathroom, in the house and the basement."),
  item(3, "HouseAge: the year sold minus the year built, set to 0 if negative (Section 3.4)."),
  item(4, "RemodelAge: the year sold minus the year of the last remodel, also set to 0 if negative."),
  item(5, "TotalPorchSF: the sum of the five deck and porch areas."),
  body(`The score did not improve (${delta("4", "3")}). For Ridge this makes sense. A linear model can already add two features by giving them equal weights, so TotalSF gives it nothing new. We kept the features anyway. After Step 5, log(TotalSF) is new information, because the log of a sum is not the sum of the logs, and tree models split on one feature at a time, so they cannot add features themselves.`),
  h2("5.4 Log of skewed area features (Step 5)"),
  body("Most area features are right-skewed like the price: many average houses and a few very large ones. In Step 5 we apply log1p, which is log(1 + x), to every area feature with a training-set skewness above 0.75. That covers 20 of our 21 area features; only GarageArea (skewness 0.18) stays as it is. We use log1p because many houses have an area of 0, for example no basement, and log(0) is undefined while log1p(0) = 0."),
  body("This step cut the CV error from 0.1512 to 0.1324, about 12% and the largest gain of any step. House prices grow roughly in proportion to size: a house twice as large tends to cost a certain percentage more, not a fixed number of dollars more. On a log-log scale a proportional relationship is a straight line, which is the shape a linear model can fit. The log also pulls in extreme values, such as one lot of 215,245 sq ft, that would otherwise drag the regression line towards them."),
  body("This also explains the small gain in Step 2. With a log target and raw areas, Ridge was fitting a curve with a straight line. The log target and the log features work as a pair."),
  h2("5.5 Removing outliers and tuning alpha (Step 6)"),
  body("Step 6 removes the two outliers from the training folds (Section 3.3) and improves CV from 0.1324 to 0.1277. We then tuned Ridge's penalty strength, alpha, over 1, 3, 10, 30 and 100. The scores formed a U shape, from 0.1303 at alpha = 1 down to 0.1277 at 10 and back up to 0.1298 at 100. A small alpha lets the model follow noise in the training houses, and a large alpha holds the weights back too much. Our first guess of 10 was already the best."),
  h2("5.6 Applying the same steps to test"),
  body("Every change has to reach the test set in exactly the same form. We wrote each step as a function, such as fill_missing_meaning or add_new_features, and call it on both files. Choices that depend on the data come from the training set only: the list of skewed features uses the training skewness, and the medians and encodings are learned inside the Pipeline."),
  h2("5.7 Most useful feature engineering"),
  body("Our answer to the first question in the brief is Step 5, the log of the skewed area features (Section 5.4). It cut the CV error by about 12%, more than all the other feature steps together, and Kaggle confirmed it: S2 and S3 differ only in this step, and the score fell from 0.13044 to 0.12253."),

  // 6
  h1("6 Models"),
  body("With the features fixed, Steps 7 to 11 compare seven models. All of them predict log price on the same five folds and train without the two outliers, so only the model changes."),
  h2("6.1 Linear models: Ridge, Lasso and ElasticNet"),
  body("A linear model predicts the log price as a weighted sum of the features. With 324 columns after one-hot encoding and many overlapping features (Section 2.2), plain linear regression can give large, unstable weights. Ridge, Lasso and ElasticNet add a penalty for large weights. Ridge penalises squared weights and shrinks all of them a little. Lasso penalises absolute weights, which can set some weights to exactly zero and so removes features. ElasticNet mixes the two. We tuned the penalty strength alpha for each model."),
  body("Lasso (0.1255) beat Ridge (0.1277). It kept 107 of the 324 columns and set the rest to zero. In a small test, we added a column of random noise next to OverallQual and GrLivArea: Ridge gave the noise a small weight (−0.0036), and Lasso set it to 0."),
  h2("6.2 Random Forest"),
  body("A decision tree predicts by asking a series of yes/no questions about the features. Table 7 shows a tree with two levels, trained on OverallQual and GrLivArea. It splits the houses into four groups and gives every house in a group the same price. A Random Forest grows many deeper trees, each on a random sample of houses and features, and averages them; ours has 300 trees."),
  ...table(7, "A two-level decision tree on log price",
    ["Group (rule)", "Mean log price", "Price"],
    [
      ["OverallQual ≤ 6.5 and GrLivArea ≤ 1378.5", "11.70", "$121k"],
      ["OverallQual ≤ 6.5 and GrLivArea > 1378.5", "11.99", "$161k"],
      ["OverallQual 6.5 to 7.5", "12.22", "$203k"],
      ["OverallQual > 7.5", "12.59", "$294k"],
    ],
    [5000, 2200, 2200]),
  body("Random Forest was our weakest model (0.1382), behind Ridge. A tree turns a smooth relationship into a staircase. After Step 5 the main relationships are close to straight lines, and every tree in the forest makes the same kind of staircase error, which averaging cannot remove."),
  h2("6.3 Boosting: XGBoost and CatBoost"),
  body("Boosting also uses trees, but builds them one after another. Each new small tree is trained on the errors that are still left, and only a small share of its prediction is added; that share is the learning rate, 0.03 in our models. Figure 5 compares the two ideas on five features, training on 80% of the houses and validating on the rest. More trees help Random Forest at first and then stop helping, while boosting keeps improving to a lower error. Boosting is best at about 140 trees here and then gets slightly worse, so the number of trees is a setting to choose as well."),
  ...figure(F("f16_bagging_boosting.png"), 5, "Bagging (Random Forest) and boosting on a five-feature hold-out experiment", 450),
  body("XGBoost uses 1,500 trees of depth 3 on the same one-hot features as the linear models. CatBoost uses 1,500 trees of depth 6 and encodes the text features itself, so it gets the Step 4 features with the text columns unchanged. Both scored about 0.123, the best of our single models."),
  h2("6.4 Support Vector Regression"),
  body("Support Vector Regression (SVR) with an RBF kernel predicts a house's price from its similarity to the training houses, measured by the distance between their scaled features. C controls how closely the model fits the training data, and gamma controls how fast similarity fades with distance. We searched a grid of both, and the best pair, C = 1 and gamma = 0.003, scored 0.1235."),
  h2("6.5 Model comparison"),
  body("Table 8 puts the seven models side by side. The top three, XGBoost, CatBoost and SVR, are within 0.0004 of each other, much less than the spread between folds (Section 7.1), so we did not pick a single winner and blended them instead (Section 6.6). CatBoost was by far the slowest, which mattered because we reran the notebook many times."),
  ...table(8, "Single models: CV RMSE and time for the full five-fold CV",
    ["Model", "Family", "CV RMSE", "Time"],
    [
      ["XGBoost", "Boosted trees", f5(cv["9"]), "4 s"],
      ["CatBoost", "Boosted trees", f5(cv["10"]), "86 s"],
      ["SVR (RBF)", "Kernel", f5(cv["11"]), "5 s"],
      ["Lasso", "Linear", f5(cv["7"]), "2 s"],
      ["ElasticNet", "Linear", f5(cv["7b"]), "3 s"],
      ["Ridge", "Linear", f5(cv["6b"]), "0.4 s"],
      ["Random Forest", "Bagged trees", f5(cv["8"]), "12 s"],
    ],
    [2700, 2700, 2000, 2000]),
  h2("6.6 Blending"),
  body("A blend averages the predictions of several models. It helps only when the models make different mistakes, so that one model's error above the true price cancels another's error below it. In the four-house example in Table 9, model B makes almost the same errors as model A, so averaging them changes little. Model C makes opposite errors, and averaging A and C cancels most of the error."),
  ...table(9, "Averaging two models on four houses (log price)",
    ["Models", "Correlation of their errors", "RMSE"],
    [
      ["A alone", "-", "0.069"],
      ["A + B (similar errors)", "+0.99", "0.066"],
      ["A + C (opposite errors)", "−1.00", "0.010"],
    ],
    [3800, 3400, 2200]),
  body("Figure 6 shows how strongly the errors of our real models correlate. Ridge, Lasso and ElasticNet correlate at 0.99, so they make almost the same mistakes. The linear models and the boosted trees correlate at about 0.88."),
  ...figure(path.join(V2, "results", "figures", "06_oof_correlation.png"), 6, "Correlation between the models' OOF errors", 460),
  body("Before looking at any blend score, we fixed a rule: take the best model of each family and keep both boosting models. That gives Lasso, XGBoost, CatBoost and SVR. We average them with equal weights, because weights tuned on the same 1,460 OOF predictions would fit their noise. The blend scores 0.1191 (Table 10). Adding every model except Random Forest makes it worse, because three almost identical linear models then hold three of the six votes."),
  ...table(10, "Blends compared after the rule was fixed",
    ["Blend", "CV RMSE"],
    [
      ["Best single model (XGBoost)", "0.12310"],
      ["XGBoost + CatBoost", "0.12099"],
      ["Lasso + XGBoost", "0.12044"],
      ["Our blend: Lasso + XGBoost + CatBoost + SVR", "0.11906"],
      ["All models except Random Forest", "0.12022"],
    ],
    [6200, 3200]),

  // 7
  h1("7 Error Analysis"),
  h2("7.1 CV and Kaggle disagree"),
  body("We submitted to Kaggle six times during Steps 1 to 12 (S1 to S6) and once more after the error analysis (S7). Each submission matches one step, so we can compare its CV score with its Kaggle score (Table 11, Figure 7). Up to S4 both fall together. After that they disagree. CatBoost (S5) beat Ridge (S4) on CV but not on Kaggle, and the blend (S6) had our best CV score but scored 0.12037 on Kaggle, worse than our first attempt (0.11745)."),
  ...table(11, "CV and Kaggle scores of every submission",
    ["Submission", "Step", "CV RMSE", "Kaggle score"],
    [
      ["S1 Ridge, raw price", "1", "0.15534", "0.14188"],
      ["S2 Log target, NA labels, new features", "4", "0.15121", "0.13044"],
      ["S3 Log of skewed areas", "5", "0.13239", "0.12253"],
      ["S4 Outliers dropped", "6", "0.12766", "0.12236"],
      ["S5 CatBoost", "10", "0.12320", "0.12310"],
      ["S6 Blend", "12", "0.11906", "0.12037"],
      ["S7 Outlier flag (final)", "13", "0.11910", "0.11624"],
      ["First attempt (for reference)", "-", "0.12117", "0.11745"],
    ],
    [4600, 1000, 1900, 1900]),
  ...figure(F("f11_cv_vs_kaggle.png"), 7, "Our CV and Kaggle scores for S1 to S7", 450),
  body("Part of this is noise. Scored fold by fold, the same Ridge model ranged from 0.105 to 0.172, and a leaderboard score is one more sample like these, so a difference of 0.001 between two submissions means little. Instead of trying more models, we looked at where our CV error came from."),
  h2("7.2 Where our errors come from"),
  body("We computed the error of every training house. Two houses dominate: the outliers 524 and 1299, which our blend priced at about 3.5 and 4 times their real price. These two errors make up 17% of the total squared error; without them, the CV RMSE would be 0.1085 instead of 0.1191. Both houses sit in fold 3, so our CV never trains on one while predicting the other, and it cannot tell us how well we predict this kind of house."),
  h2("7.3 Test house 2550"),
  body("The test set has a third house of this kind. Id 2550 has 5,095 sq ft, the top quality rating of 10 and a Partial sale. Its two look-alikes sold for $160,000 and $184,750, but every submission from S1 to S6 predicted between $565,000 and $891,000 for it (Figure 8). Our first attempt predicted $491,000. The first attempt's CatBoost was trained with the outliers and predicted $306,000, while the same model trained without them predicted $618,000. One house can matter this much: if the true price of house 2550 is close to its look-alikes', a log error of about 1.3 on this house alone adds about 0.005 to an RMSE of 0.116 over all 1,459 test houses, similar to the gap between S6 and S7."),
  ...figure(F("f12_id2550.png"), 8, "Our predictions for test house 2550; the red lines are the prices of its two look-alikes", 420),
  h2("7.4 Step 13: keep the outliers and flag them"),
  body("In Step 13 we keep the two outliers in training and add one feature, HugePartial. It is 1 for houses over 4,000 sq ft sold as Partial sales and 0 for all other houses, so a model can learn a separate adjustment for this kind of sale and leave normal houses alone. We compute the flag from the raw area (see the outlier problem in Section 9). Step 13 also includes the garage-year fix from Section 3.4."),
  body("Normal CV cannot test this change, because both outliers sit in the same fold. We designed our own check: hold out one outlier, train on all other houses, predict it, and repeat for the other one (Figure 9). With the earlier method, the blend predicted 3.6 to 4.4 times the true price. With Step 13, it predicted 1.3 to 1.9 times."),
  ...figure(F("f13_loo.png"), 9, "Leave-one-outlier-out check: true price, earlier method and Step 13", 400),
  body("We refitted the same four models with the same settings and equal weights. The CV score stayed at 0.1191, since CV cannot see this effect. The prediction for house 2550 fell from $643,000 to $260,000, and the other test houses changed by a median of 0.25%. We submitted it once, as S7. It scored 0.11624, our best score and better than our first attempt."),
  h2("7.5 Ideas we tried and did not keep"),
  body("We tried four more ideas on the Step 13 features (Table 12). None beat our blend on CV, so we did not submit them. Step 13 itself came from the leaderboard, so we submitted it only once and stopped there, to avoid tuning our model to the public test houses."),
  ...table(12, "Ideas we tried after Step 13",
    ["Idea", "CV RMSE"],
    [
      ["Our Step 13 blend", "0.11910"],
      ["Stacking: a model learns the blend weights", "0.12260"],
      ["The first attempt's recipe (Lasso + CatBoost)", "0.12026"],
      ["Lasso with the first attempt's quality features", "0.12606"],
      ["Our blend with XGBoost averaged over 3 seeds", "0.11911"],
    ],
    [6200, 3200]),

  // 8
  h1("8 Results"),
  h2("8.1 Final result"),
  body("Our final submission, S7, scores 0.11624 and placed us at rank 257 (Appendix A). Over our seven submissions the score fell from 0.14188 to 0.11624 (Table 11), and S7 also beats our first attempt (0.11745). Appendix B lists the CV score of every step."),
  h2("8.2 Best improvement between submissions"),
  body("The brief asks us to \"describe the case your team has obtained the best improvement (not absolute performance)\". Our biggest improvement on Kaggle was from S1 to S2: 0.14188 to 0.13044, a drop of 0.0114 (8.1%). S2 combined three changes, namely predicting the log price, labelling missing values and the new features, and our CV shows that the log target gave most of the gain (−0.0033 of −0.0041 on CV). On Kaggle the same changes gained almost three times as much as on CV; with one score per submission, we cannot tell how much of that is noise (Section 7.1)."),
  body("We learned the most from S6 to S7 (0.12037 to 0.11624). That gain came from looking at individual errors. We found a blind spot in our CV, worked out why one test house was predicted four times too high and fixed it with one feature (Section 7)."),

  // 9
  h1("9 Problems Found and How We Responded"),
  body("Some problems are described where they happened: the \"None\" values in Section 3.1, the garage year in Section 3.4 and the gap between CV and Kaggle in Section 7. Table 13 lists the others. Several were silent: the code ran without an error, and only a score that did not move showed that something was wrong."),
  ...table(13, "Other problems we met during the project",
    ["Problem", "How we noticed", "What we did"],
    [
      ["Our outlier rule stopped working", "Step 6 scored exactly the same as Step 5", "Built the mask from the raw data; added a test"],
      ["CatBoost could not be copied for each fold", "scikit-learn's clone raised an error", "Passed the text features as a tuple"],
      ["Best SVR settings on the edge of the grid", "The best C and gamma were at the end of the grid", "Widened the grid twice; checked beyond the corner"],
      ["Rerunning a step duplicated its result", "The results table had repeated steps", "The record function now replaces the old row"],
      ["A test overwrote our real results file", "cv_results.csv changed after the tests ran", "Tests now write to a temporary folder"],
    ],
    [3000, 3200, 3200], ["l", "l", "l"]),
  body("The outlier rule was the clearest example. It tested GrLivArea > 4000 inside the CV function, but Step 5 had already replaced GrLivArea with its log, whose maximum is 8.64, so no house matched and nothing was removed. The code ran without complaint, and we only noticed because Step 6 scored exactly the same as Step 5, which cannot happen when two houses leave the training data. We now compute the outlier mask once from the raw data, and an automated test fails if the two steps ever score the same again."),

  // 10
  h1("10 Limitations"),
  item(1, "We made every choice with the same five folds, so our CV scores are slightly optimistic. A fresh split, or repeated CV with other seeds, would give a fairer estimate."),
  item(2, "The HugePartial flag rests on two training houses. It fixed house 2550, but we cannot be sure it would help with other unusual sales."),
  item(3, "S2 bundled three changes, so we cannot split its Kaggle gain between them. Next time we would submit after every step."),
  item(4, "Both outliers fell in the same fold, which hid the problem in Section 7. Spreading rare houses across the folds would have shown it earlier."),
  item(5, "We used fixed, common settings for XGBoost and CatBoost and tuned only the linear models and SVR. A careful search might improve the boosting models a little."),

  // 11
  h1("11 Conclusion"),
  body("We rebuilt a Kaggle solution step by step and measured each decision, which took our Kaggle score from 0.142 to 0.116. Changing one thing at a time told us what each idea was worth, including the ideas that did nothing. The log of the skewed areas gave our largest gain because it matched the shape of the data. Our best CV score still hid a blind spot, and looking at the errors of individual houses helped us more than trying more models."),
];

// ---------- appendices ----------
const APPX = [
  pageBreak(),
  h1("Appendix A. Kaggle result"),
  ...figure(F("kaggle_screenshot.png"), 10, "Our team on the Kaggle leaderboard: rank 257, score 0.11624", 620),
  h1("Appendix B. Experiment log"),
  body("Table 14 lists every step in the notebook. Steps 1 to 6 are compared with the step before, Steps 7 to 11 with Ridge after tuning (Step 6b), the blend with the best single model (Step 9) and Step 13 with the blend."),
  ...table(14, "Every step and its CV RMSE",
    ["Step", "Change", "CV RMSE", "Compared with"],
    [
      ["0", "Predict the mean price", f5(cv["0"]), "-"],
      ["1", "Ridge, raw price", f5(cv["1"]), `${delta("1", "0")} vs Step 0`],
      ["2", "Predict log(SalePrice)", f5(cv["2"]), `${delta("2", "1")} vs Step 1`],
      ["3", "Label NA by meaning; MSSubClass as text", f5(cv["3"]), `${delta("3", "2")} vs Step 2`],
      ["4", "New combined features", f5(cv["4"]), `${delta("4", "3")} vs Step 3`],
      ["5", "log1p of skewed area features", f5(cv["5"]), `${delta("5", "4")} vs Step 4`],
      ["6", "Drop 2 outliers from training folds", f5(cv["6"]), `${delta("6", "5")} vs Step 5`],
      ["6b", "Ridge alpha tuned (best: 10)", f5(cv["6b"]), "no change"],
      ["7", "Lasso", f5(cv["7"]), `${delta("7", "6b")} vs Step 6b`],
      ["7b", "ElasticNet", f5(cv["7b"]), `${delta("7b", "6b")} vs Step 6b`],
      ["8", "Random Forest (300 trees)", f5(cv["8"]), `${delta("8", "6b")} vs Step 6b`],
      ["9", "XGBoost", f5(cv["9"]), `${delta("9", "6b")} vs Step 6b`],
      ["10", "CatBoost", f5(cv["10"]), `${delta("10", "6b")} vs Step 6b`],
      ["11", "SVR", f5(cv["11"]), `${delta("11", "6b")} vs Step 6b`],
      ["12", "Blend of Lasso, XGBoost, CatBoost, SVR", f5(cv["12"]), `${delta("12", "9")} vs Step 9`],
      ["13", "Keep outliers + HugePartial flag; same blend", f5(cv["13"]), `${delta("13", "12")} vs Step 12`],
    ],
    [900, 4600, 1600, 2400], ["c", "l", "c", "l"]),
  h1("Appendix C. Code excerpts"),
  body("Kaggle notebook: [link]. Repository with the notebook and our study notes: [link]. The excerpts below are copied from the notebook."),
  h3("C.1 Cross-validation (Part 2)"),
  ...code(sourceLines("part2_baseline.py", "def cv_evaluate")),
  h3("C.2 Missing values by meaning (Part 3)"),
  ...code(sourceLines("part3_features.py", "def fill_missing_meaning")),
  h3("C.3 Blend (Part 5)"),
  ...code(sourceLines("part5_blend.py", "blend_names = ", 3)),
  h3("C.4 Step 13 helpers (Part 6)"),
  ...code([...sourceLines("part6_error_analysis.py", "def clean_years"), "",
           ...sourceLines("part6_error_analysis.py", "def add_huge_partial_flag")]),
  h1("Appendix D. Individual answers"),
  ...["[Name 1]", "[Name 2]"].flatMap((who) => [
    h3(who),
    body([{ b: "Main role: " }, "[...]"]),
    body([{ b: "Main difficulties (team / own): " }, "[...]"]),
    body([{ b: "What I learned: " }, "[...]"]),
  ]),
];

// ---------- document ----------
const runningHeader = new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
  children: [run("House Price Prediction", { italics: true, size: 18 })] })] });
const pageFooter = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] });

const doc = new Document({
  creator: "[team name]",
  title: "House Price Prediction",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 30, bold: true },
        paragraph: { spacing: { before: 300, after: 100 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true },
        paragraph: { spacing: { before: 180, after: 60 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, italics: true },
        paragraph: { spacing: { before: 140, after: 60 }, keepNext: true, outlineLevel: 2 } },
      { id: "TOC1", name: "toc 1", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 22, bold: true }, paragraph: { spacing: { before: 40, after: 0 } } },
      { id: "TOC2", name: "toc 2", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 22 }, paragraph: { spacing: { before: 0, after: 0 }, indent: { left: 440 } } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 },
      margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } }, titlePage: true },
    headers: { default: runningHeader, first: new Header({ children: [new Paragraph({ children: [] })] }) },
    footers: { default: pageFooter, first: pageFooter },
    children: [...FRONT, ...BODY, ...APPX],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  const out = path.join(__dirname, "House_Prices_Report.docx");
  fs.writeFileSync(out, buffer);
  console.log("wrote", out);
});
