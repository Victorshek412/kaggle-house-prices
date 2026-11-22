// Builds report/House_Prices_Report_draft.docx: two-column layout modelled on the course sample,
// Times New Roman 12. Run from the repo root after `python report/make_figures.py`:
//     node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, Header, Footer,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, PageNumber, SectionType,
} = require("docx");

const FONT = "Times New Roman";
const BLUE = "1F3FBF";
const FIGS = path.join(__dirname, "figures");
const V2FIGS = path.join(__dirname, "..", "v2", "results", "figures");
const COL_DXA = 4693;       // one column: (A4 width - 2 x 0.75 in margins - 0.25 in gap) / 2
const COL_PX = 300;         // image width inside one column

// ---------- helpers ----------
const run = (text, o = {}) => new TextRun({ text, font: FONT, size: 24, ...o });

// body("text") or body(["plain ", {b: "bold"}, {i: "italic"}])
function body(parts, o = {}) {
  const list = Array.isArray(parts) ? parts : [parts];
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 100, line: 252 },
    ...o,
    children: list.map((p) => typeof p === "string" ? run(p)
      : p.b ? run(p.b, { bold: true }) : run(p.i, { italics: true })),
  });
}

function item(label, text) { // sample-style numbered item: "1. (Label) text"
  return body([`${label[0]} `, { b: `(${label[1]})` }, ` ${text}`]);
}

const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const h3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)] });

function caption(kind, n, text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 60, after: 180 },
    children: [run(`${kind} ${n}. `, { bold: true, size: 20 }), run(text, { italics: true, size: 20 })],
  });
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
}

function figure(file, n, text, widthPx = COL_PX) {
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

function placeholderFigure(n, text, note) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 100, after: 0 }, keepNext: true,
      border: ["top", "bottom", "left", "right"].reduce((o, k) =>
        ({ ...o, [k]: { style: BorderStyle.DASHED, size: 6, color: "999999", space: 6 } }), {}),
      children: [run(note, { italics: true, color: "777777", size: 20 })],
    }),
    caption("Figure", n, text),
  ];
}

// booktabs-style table: rules above and below the header and at the bottom, no vertical lines
function table(n, captionText, headers, rows, widths) {
  const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
  const thick = { style: BorderStyle.SINGLE, size: 12, color: "000000" };
  const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
  const make = (cells, i, last) => new TableRow({
    children: cells.map((text, j) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA },
      margins: { top: 30, bottom: 30, left: 60, right: 60 },
      borders: { left: none, right: none,
        top: i === 0 ? thick : i === 1 ? thin : none,
        bottom: last ? thick : none },
      children: [new Paragraph({
        keepNext: true,
        alignment: /^[-+$0-9. s]+$/.test(all[1][j]) ? AlignmentType.CENTER : AlignmentType.LEFT,
        children: [run(text, { size: 20, bold: i === 0 })],
      })],
    })),
  });
  const all = [headers, ...rows];
  return [
    new Table({
      width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
      columnWidths: widths,
      borders: { insideHorizontal: none, insideVertical: none, top: none, bottom: none, left: none, right: none },
      rows: all.map((r, i) => make(r, i, i === all.length - 1)),
    }),
    caption("Table", n, captionText),
  ];
}

const rule = () => new Paragraph({
  spacing: { after: 120 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: "000000", space: 1 } },
  children: [],
});

// ---------- title block (one column) ----------
const titleBlock = [
  rule(),
  new Paragraph({ spacing: { before: 120, after: 120 }, children: [run("House Price Prediction", { bold: true, size: 40 })] }),
  new Paragraph({ spacing: { after: 60 }, children: [run("[Team name]: [Name] ([Student ID]), [Name] ([Student ID])", { bold: true })] }),
  new Paragraph({ spacing: { after: 120 }, children: [run("Kaggle competition: House Prices, Advanced Regression Techniques.  Final score: 0.11624", { size: 18 })] }),
  rule(),
];

// ---------- body (two columns) ----------
const F = (name) => path.join(FIGS, name);
const B = [
  h1("1. Abstract"),
  body("We took part in the Kaggle competition House Prices: Advanced Regression Techniques. The task is to predict the sale price of 1,459 houses in Ames, Iowa, from 79 features. Kaggle scores a submission by the RMSE between the log of the predicted price and the log of the true price, and our final submission scores 0.11624. We built the solution in 13 steps. We changed one thing at a time and scored every step with the same five-fold cross-validation (CV), so we always knew what each decision was worth."),
  body([{ b: "Keywords" }, " - regression, feature engineering, cross-validation, gradient boosting, ensemble, error analysis"]),
  body("In this report, we introduce:"),
  body("(1) what the data looks like and what it told us to do;", { spacing: { after: 0 } }),
  body("(2) how we validated, engineered features, compared models and blended them;", { spacing: { after: 0 } }),
  body("(3) how our models did on the Kaggle leaderboard, and how an error analysis fixed a problem our CV could not see;", { spacing: { after: 0 } }),
  body("(4) our answers to the two questions in the brief: the most useful feature engineering and the best improvement."),

  h1("2. Dataset"),
  body("Kaggle gives us 1,460 training houses with prices and 1,459 test houses without. Each house has 79 features: 36 numeric ones, such as living area and year built, and 43 text ones, such as neighbourhood and garage type."),
  h2("2.1. Target"),
  body("SalePrice is right-skewed, with a skewness of 1.88. The middle half of the houses sold for $130,000 to $214,000, while the most expensive one sold for $755,000. After a log transform the skewness drops to 0.12 (Figure 1). So we predict log(SalePrice), which is also the scale Kaggle uses for scoring."),
  ...figure(F("f01_target.png"), 1, "SalePrice before and after the log transform"),
  h2("2.2. Missing values"),
  body("The data description tells us that NA means two different things. For many features, NA means the house does not have the item: PoolQC is missing for 1,453 houses because they have no pool. For others, such as LotFrontage (259 missing), the value was simply not recorded (Figure 2). We also found 15 features with gaps only in the test set, such as MSZoning and GarageCars, so our preprocessing has to cope with gaps it never sees in training."),
  ...figure(F("f02_missing.png"), 2, "The 12 training features with the most NA values"),
  body("One detail caught us out. By default, pandas reads the text \"None\" as missing, but MasVnrType = None is a real category that means no masonry veneer. With the default settings 872 houses looked unknown. We read the files with keep_default_na=False, and only 8 remain unknown."),
  h2("2.3. Outliers and correlation"),
  body("Two houses, Ids 524 and 1299, have over 4,000 sq ft of living area but sold for less than $300,000 (Figure 3). Both were Partial sales, which means the homes were not finished when they were last assessed. The two other houses of that size sold for $745,000 and $755,000, so we only treat the two cheap ones as outliers."),
  ...figure(F("f03_outliers.png"), 3, "Living area against price; the circled houses are the outliers"),
  body("Among the numeric features, OverallQual (0.79) and GrLivArea (0.71) correlate most with price (Figure 4). Correlation cannot see the 43 text features, so we used it as a map and did not remove any feature because of it."),
  ...figure(F("f04_correlation.png"), 4, "Numeric features with the highest correlation to SalePrice"),

  h1("3. Method"),
  h2("3.1. Overall"),
  body("Figure 5 shows our pathway. First, we set up a fair way to score models. Then we improved the features one step at a time with a fixed Ridge model, compared seven models on the final features and blended the best ones. After submitting, we compared our CV with the leaderboard and made one last change (Step 13)."),
  ...figure(F("f05_pathway.png"), 5, "Our pathway from data to the final submission", 270),
  h2("3.2. First attempt"),
  body("Our first submission, on 9 November, came from a pipeline that we generated with an AI coding assistant (Codex). It ran 27 experiments, blended ElasticNet with CatBoost, and scored 0.1212 on CV and 0.11745 on Kaggle. However, we could not explain why each feature or setting was there. So we rebuilt everything in one notebook that we can explain line by line. For the rebuild we used another assistant (Claude) as a tutor, and we wrote study notes for every part before moving on. Our AI-use statement gives the details."),
  h2("3.3. Validation set up"),
  body("We use five-fold CV with fixed folds (KFold with shuffle and random_state=42). A model trained on 1,168 houses predicts the other 292, and each house takes its turn (Figure 6). Every house therefore gets one out-of-fold (OOF) prediction, and the RMSE of these predictions is our CV score. Our test prediction is the average of the five fold models."),
  ...figure(F("f06_folds.png"), 6, "Which fold validates each of the first 20 houses"),
  body("To avoid data leakage, we fit the median imputation, scaling and one-hot encoding inside a scikit-learn Pipeline in every fold. For example, the median LotFrontage is 69 over all houses but 70 over the training part of fold 1. Predicting the mean price for every house gives a CV RMSE of 0.408 (Step 0), and Ridge on the raw price gives 0.155 (Step 1)."),

  h2("3.4. Feature engineering"),
  h3("3.4.1. Feature construction."),
  body("We keep the same Ridge model (alpha = 10) and change one thing per step (Table 1). Our notebook prints the change after every step (Figure 7)."),
  ...table(1, "Feature engineering steps (Ridge, five-fold CV RMSE)",
    ["Step", "Change", "CV RMSE"],
    [
      ["1", "Raw price (baseline)", "0.15534"],
      ["2", "Predict log(SalePrice)", "0.15206"],
      ["3", "Label NA by meaning; MSSubClass as text", "0.15106"],
      ["4", "TotalSF, TotalBathrooms, ages, TotalPorchSF", "0.15121"],
      ["5", "log1p of 20 skewed area features", "0.13239"],
      ["6", "Drop 2 outliers from training folds", "0.12766"],
    ],
    [500, 3093, 1100]),
  ...figure(F("f14_console.png"), 7, "Notebook output for Steps 0 to 6"),
  body("In Step 3, we replace NA with a label such as NoGarage only when another feature confirms it: GarageType becomes NoGarage only if GarageArea is 0. We also turn MSSubClass into text, because its values (20, 60, 190 and so on) are codes for house types. In Step 5, we take log1p of every area feature with a training skewness above 0.75, which covers 20 of our 21 area features. In Step 6, we remove the two outliers from the training rows of each fold, but they are still predicted and scored. Figure 8 shows how much larger the Step 5 drop is than the others."),
  ...figure(F("f07_staircase.png"), 8, "CV RMSE after each feature step"),
  h3("3.4.2. Difficulties & trials on improvement."),
  item(["1.", "New features did nothing"], "Step 4 left the CV score unchanged (+0.00014). Ridge can already add two features together by giving them equal weights, so TotalSF gave it nothing new. We kept these features because log(TotalSF) adds new information after Step 5, and tree models cannot add features by themselves."),
  item(["2.", "Our outlier rule stopped working"], "Our first rule checked GrLivArea > 4000 inside the CV function, and Step 6 scored exactly the same as Step 5. Step 5 had replaced GrLivArea with its log, whose maximum is 8.64, so no house matched. We now compute the outlier mask once from the raw data, and a test fails if the two steps ever score the same again."),
  item(["3.", "Labelling NA gave little"], "Step 3 only improved CV by 0.0010. The Pipeline already turned missing text into a \"Missing\" category, which the model used in the same way as our labels."),
  h3("3.4.3. Reflection."),
  body("Step 5 gave the largest gain (-0.0188, about 12%). Step 2 alone gave only -0.0033 because the features were still on their raw scale, so the log target and the log features work as a pair. Finally, we tuned Ridge's alpha. The CV score forms a U shape (Figure 9): a small alpha overfits and a large one underfits. Our first guess of 10 was already the best."),
  ...figure(F("f08_alpha.png"), 9, "Ridge alpha against CV RMSE"),

  h2("3.5. Model comparison"),
  h3("3.5.1. Training."),
  body("With the features fixed, we compared seven models on the same folds, all predicting log price and trained without the two outliers (Table 2, Figure 10). The linear models and SVR use the log features and one-hot encoding. CatBoost uses the features from Step 4 and handles the text features itself."),
  ...table(2, "Single models (CV RMSE and time for the full CV)",
    ["Model", "Family", "CV RMSE", "Time"],
    [
      ["XGBoost", "Boosted trees", "0.12310", "4 s"],
      ["CatBoost", "Boosted trees", "0.12320", "86 s"],
      ["SVR (RBF)", "Kernel", "0.12349", "5 s"],
      ["Lasso", "Linear", "0.12546", "2 s"],
      ["ElasticNet", "Linear", "0.12565", "3 s"],
      ["Ridge", "Linear", "0.12766", "0.4 s"],
      ["Random Forest", "Bagged trees", "0.13822", "12 s"],
    ],
    [1393, 1400, 1000, 900]),
  ...figure(F("f09_models.png"), 10, "CV RMSE of the seven models"),
  h3("3.5.2. Difficulties & trials on improvement."),
  item(["1.", "SVR settings at the edge"], "The best C and gamma kept landing on the edge of our grid, so we widened the grid twice. We then checked four pairs just outside the final corner (C = 1, gamma = 0.003), and all four scored worse."),
  item(["2.", "CatBoost could not be copied"], "Our CV function copies the model for each fold with scikit-learn's clone, which failed for CatBoost when we passed the text features as a list. Passing them as a tuple fixed it."),
  h3("3.5.3. Reflection."),
  body("Random Forest was the weakest model, behind even Ridge. Averaging many deep trees removes random errors, but after Step 5 the main relationships are close to straight lines, and every tree approximates them with the same steps. Boosting adds small trees one after another, and each tree fixes the error that is left, so XGBoost and CatBoost came first. Lasso beat Ridge because it kept only 107 of the 324 columns and set the rest to zero."),
  body("Figure 16 shows the difference between bagging and boosting in a small experiment with five features, where we train on 80% of the houses and validate on the rest. More trees help Random Forest at first but soon stop helping, while boosting keeps improving as it adds trees and reaches a lower error. Boosting is best at about 140 trees here and then gets slightly worse, so the number of trees is a setting we have to choose too."),
  ...figure(F("f16_bagging_boosting.png"), 16, "Bagging and boosting on a five-feature hold-out experiment"),

  h2("3.6. Blending"),
  body("A blend helps only when its members make different mistakes. Figure 11 shows how strongly the models' OOF residuals correlate. Ridge, Lasso and ElasticNet correlate at 0.99 or more, so they make almost the same errors. The linear models and the boosted trees correlate at about 0.88."),
  ...figure(path.join(V2FIGS, "06_oof_correlation.png"), 11, "Correlation between the models' OOF residuals", 280),
  body("Before we looked at any blend score, we fixed a rule: take the best model of each family and keep both boosting models. This gives Lasso, XGBoost, CatBoost and SVR. We average them with equal weights, because weights tuned on the same 1,460 OOF predictions would fit their noise. The blend scores 0.1191 (Table 3). Adding every model except Random Forest makes it worse, because three almost identical linear models then hold three of the six votes."),
  ...table(3, "Blends compared after the rule was fixed",
    ["Blend", "CV RMSE"],
    [
      ["Best single model (XGBoost)", "0.12310"],
      ["XGBoost + CatBoost", "0.12099"],
      ["Lasso + XGBoost", "0.12044"],
      ["Our blend (4 models)", "0.11906"],
      ["All except Random Forest", "0.12022"],
    ],
    [3293, 1400]),

  h1("4. Test on Kaggle"),
  h2("4.1. Leaderboard results"),
  body("Table 4 and Figure 13 put our CV scores next to the Kaggle scores. Up to S4, both go down together. After that they disagree. CatBoost (S5) beat Ridge (S4) on CV but not on Kaggle, and our blend (S6) scored worse than our first attempt (0.12037 against 0.11745)."),
  ...table(4, "CV and Kaggle scores of every submission",
    ["Submission", "CV", "Kaggle"],
    [
      ["S1 Ridge, raw price", "0.15534", "0.14188"],
      ["S2 Log target, NA, features", "0.15121", "0.13044"],
      ["S3 Log of skewed areas", "0.13239", "0.12253"],
      ["S4 Outliers dropped", "0.12766", "0.12236"],
      ["S5 CatBoost", "0.12320", "0.12310"],
      ["S6 Blend", "0.11906", "0.12037"],
      ["S7 Step 13", "0.11910", "0.11624"],
      ["First attempt", "0.12117", "0.11745"],
    ],
    [2693, 1000, 1000]),
  ...placeholderFigure(12, "Our team on the Kaggle leaderboard", "[Insert screenshot: team name and score 0.11624]"),
  ...figure(F("f11_cv_vs_kaggle.png"), 13, "Our CV and Kaggle scores for S1 to S7"),
  body("Some of this is noise. When we scored the five folds separately, the same Ridge model ranged from 0.105 to 0.172. A leaderboard score is one more sample of this kind, so a difference of 0.001 between two submissions means little. To find out whether anything real was behind the gap, we looked at where our CV error came from."),
  h2("4.2. Error analysis"),
  body("Two houses dominate our CV error: the outliers 524 and 1299. Our blend priced them at about 3.5 and 4 times their real value (log errors -1.25 and -1.40). These two errors make up 17% of the total squared error; without them, our CV RMSE would be 0.1085 instead of 0.1191. Both houses also sit in fold 3, so our CV never trains on one of them while predicting the other."),
  body("The test set has a third house of this kind. Id 2550 has 5,095 sq ft, the top quality rating of 10, and a Partial sale. Since Steps 6 to 12 removed its two look-alikes from training, our models predicted $565,000 to $763,000 for it (Figure 14). Our first attempt predicted $491,000. Its CatBoost was trained with the outliers and predicted $306,000, while the same model without them predicted $618,000."),
  ...figure(F("f12_id2550.png"), 14, "Our predictions for test house Id 2550"),
  h2("4.3. Step 13"),
  body("In Step 13, we keep the two outliers in training and add a feature, HugePartial, which is 1 for houses over 4,000 sq ft sold as Partial sales. A model can then learn a separate adjustment for this kind of sale without changing anything for other houses. We compute the flag from the raw area, as in Step 6. We also found a garage built in 2207 in the test set (Id 2593), so we treat garage years later than the sale year plus one as missing."),
  body("Normal CV cannot test this change, so we designed our own check. We held out one outlier, trained on all other houses and predicted it, and then did the same for the other one (Figure 15). With the earlier method, the blend was 3.6 to 4.4 times too high. With Step 13, it was 1.3 to 1.9 times too high."),
  ...figure(F("f13_loo.png"), 15, "Leave-one-outlier-out check"),
  body("We refitted the same four models with the same settings and equal weights. The CV score stayed at 0.1191, because CV cannot see this effect. The prediction for Id 2550 fell from $643,000 to $260,000, and the other test houses changed by a median of 0.25%. We submitted it once, as S7, and it scored 0.11624, our best score and better than our first attempt."),
  h2("4.4. Reflection"),
  body("We also tried four other ideas on the Step 13 features, and none of them beat our blend (Table 5). Step 13 came from looking at the leaderboard, so we submitted it only once and did not try variants against the leaderboard."),
  ...table(5, "Ideas we tried and did not keep",
    ["Idea", "CV RMSE"],
    [
      ["Our Step 13 blend", "0.11910"],
      ["Stacking with learned weights", "0.12260"],
      ["First attempt's recipe (Lasso + CatBoost)", "0.12026"],
      ["Lasso + first attempt's quality features", "0.12606"],
      ["Blend with 3-seed XGBoost", "0.11911"],
    ],
    [3393, 1300]),

  h1("5. Answers to the questions"),
  h2("5.1. Most useful feature engineering"),
  body("Taking log1p of the skewed area features (Step 5) was the most useful. It cut our CV error from 0.1512 to 0.1324, about 12%, and Kaggle agreed (0.13044 to 0.12253). House prices grow roughly in proportion to size, so log price is close to a straight line in log area, which is the shape a linear model can fit. The log also shrinks a few very large values, such as huge lots, that would otherwise pull the regression line towards them."),
  h2("5.2. Best improvement"),
  body("Our biggest improvement on Kaggle was from S1 to S2, from 0.14188 to 0.13044 (-0.0114, or 8.1%). S2 combined three changes: predicting the log price, labelling missing values, and the new features. Our CV shows that the log target gave most of that gain (-0.0033 of -0.0041). The improvement we learned most from was S6 to S7 (0.12037 to 0.11624), which came from the error analysis in Section 4."),

  h1("6. General limitation"),
  item(["1.", "One CV for every choice"], "We made all our choices with the same five folds, so our CV scores are slightly optimistic."),
  item(["2.", "Two examples"], "Step 13 learns from only two training houses."),
  item(["3.", "Bundled submission"], "S2 contained three changes, so we cannot split its Kaggle gain. Next time, we would submit after every step and spread rare houses across different folds."),

  h1("7. Summary"),
  body("In this project, we rebuilt a Kaggle solution step by step and measured every decision. Our CV error fell from 0.155 to 0.119, and our Kaggle score fell from 0.142 to 0.116. Log-transforming the skewed areas gave the largest gain, boosted trees were the best single models, and a blend of models that make different mistakes improved the score again. The most useful lesson came at the end: the leaderboard showed us a blind spot in our CV, and looking at individual errors fixed it better than trying more models."),

  h1("Appendix"),
  body([{ b: "A. Notebook. " }, "Kaggle notebook: [link]. Repository with the notebook and our study notes: [link]."]),
  body([{ b: "B. Individual answers. " }, "[Each member answers: main role; main difficulties for the team and for themselves; what they learned.]"]),
];

// ---------- document ----------
const runningHeader = new Header({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [run("House Price Prediction", { italics: true, size: 18 })] })] });
const pageFooter = new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18, bold: true })] })] });
const margins = { top: 1080, right: 1080, bottom: 1080, left: 1080 };

const doc = new Document({
  creator: "[team name]",
  title: "House Price Prediction",
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true },
        paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true },
        paragraph: { spacing: { before: 160, after: 60 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, italics: true, color: BLUE },
        paragraph: { spacing: { before: 120, after: 40 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  sections: [
    {
      properties: { page: { margin: margins }, titlePage: true },
      headers: { default: runningHeader, first: new Header({ children: [new Paragraph({ children: [] })] }) },
      footers: { default: pageFooter, first: pageFooter },
      children: titleBlock,
    },
    {
      properties: { type: SectionType.CONTINUOUS, page: { margin: margins },
        column: { count: 2, space: 360, equalWidth: true } },
      headers: { default: runningHeader },
      footers: { default: pageFooter },
      children: B,
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  const out = path.join(__dirname, "House_Prices_Report_draft.docx");
  fs.writeFileSync(out, buffer);
  console.log("wrote", out);
});
