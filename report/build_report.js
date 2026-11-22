// Builds report/House_Prices_Report_draft.docx (Times New Roman 12).
// Run from the repo root:  node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType, Footer,
  PageNumber, PageBreak, LevelFormat,
} = require("docx");

const FIG = path.join(__dirname, "..", "v2", "results", "figures");
const FONT = "Times New Roman";
const CONTENT_WIDTH_DXA = 9026; // A4 width minus two 1-inch margins

// ---------- helpers ----------
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 276 },
    alignment: AlignmentType.JUSTIFIED,
    ...opts,
    children: [new TextRun({ text, font: FONT, size: 24 })],
  });
}

function placeholder(text) {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    alignment: AlignmentType.CENTER,
    border: {
      top: { style: BorderStyle.SINGLE, size: 6, color: "999999", space: 4 },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: "999999", space: 4 },
      left: { style: BorderStyle.SINGLE, size: 6, color: "999999", space: 4 },
      right: { style: BorderStyle.SINGLE, size: 6, color: "999999", space: 4 },
    },
    children: [new TextRun({ text, font: FONT, size: 24, italics: true, color: "666666" })],
  });
}

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text })] });
}

function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text })] });
}

function caption(text) {
  return new Paragraph({
    spacing: { before: 60, after: 200 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, font: FONT, size: 22, italics: true })],
  });
}

function figure(file, widthPx, captionText) {
  const full = path.join(FIG, file);
  const { width, height } = pngSize(full);
  const h = Math.round((widthPx * height) / width);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 0 },
      keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(full), transformation: { width: widthPx, height: h } })],
    }),
    caption(captionText),
  ];
}

function table(headers, rows, widths, captionText) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, w, header) => new TableCell({
    borders,
    width: { size: w, type: WidthType.DXA },
    shading: header ? { fill: "E7E6E6", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({
      keepNext: true, // keep the whole table on one page
      children: [new TextRun({ text, font: FONT, size: 22, bold: header })],
    })],
  });
  const total = widths.reduce((a, b) => a + b, 0);
  return [
    new Table({
      width: { size: total, type: WidthType.DXA },
      columnWidths: widths,
      rows: [
        new TableRow({ tableHeader: true, children: headers.map((t, i) => cell(t, widths[i], true)) }),
        ...rows.map((r) => new TableRow({ children: r.map((t, i) => cell(t, widths[i], false)) })),
      ],
    }),
    caption(captionText),
  ];
}

function reference(text) {
  return new Paragraph({
    spacing: { after: 100, line: 276 },
    indent: { left: 360, hanging: 360 },
    children: [new TextRun({ text, font: FONT, size: 24 })],
  });
}

// ---------- content ----------
const children = [
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "House Prices: Advanced Regression Techniques", font: FONT, size: 32, bold: true })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "Kaggle competition report", font: FONT, size: 26 })],
  }),
  ...[
    "Team: [team name]",
    "Members: [name, student ID]   [name, student ID]",
    "Course: [course code and name]",
    "Final Kaggle score: 0.11624 (submission s7_outlier_flag.csv)",
  ].map((t) => new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: t, font: FONT, size: 24 })],
  })),
  new Paragraph({ spacing: { after: 120 }, children: [] }),

  h1("Summary"),
  body("Our final Kaggle score is 0.11624, the root mean squared error (RMSE) between the log of our predicted prices and the log of the true prices. We got there in 13 steps. Each step changed one thing and was scored with the same five-fold cross-validation (CV), so we knew what a decision was worth before we kept it."),
  h2("Which feature engineering effort was most useful?"),
  body("Taking log1p of the skewed area columns (Step 5). This one change lowered our CV error from 0.1512 to 0.1324, about 12%, more than any other feature step. House prices grow roughly in proportion to size: a house twice as big costs roughly twice as much, rather than a fixed amount more. On a log scale that relationship is close to a straight line, which is the shape a linear model can fit. The log also shrinks the few very large values that would otherwise pull the regression line towards them. Section 5 explains this step in detail."),
  h2("Which submission gave the best improvement?"),
  body("Our biggest jump on the leaderboard was from S1 to S2, from 0.14188 to 0.13044 (a drop of 0.0114, or 8.1%). S2 contained three changes at once: predicting log(SalePrice) instead of the raw price, giving missing values their real meaning, and adding combined features. Our CV results for these three steps show that most of the gain came from predicting the log price. The improvement we learned the most from came at the end. Our blend (S6) scored worse than our own first attempt, and working out why led us to a blind spot in our CV. Fixing it took the score from 0.12037 (S6) to 0.11624 (S7). Section 8 describes how."),
  placeholder("[Insert screenshot of the Kaggle leaderboard showing our team name and final score 0.11624]"),

  h1("1  Introduction"),
  body("The competition asks for the sale prices of 1,459 houses in Ames, Iowa. Each house is described by 79 columns: 36 numeric ones such as living area and year built, and 43 text ones such as neighbourhood and garage type. We learn from 1,460 houses whose prices are given. The data was compiled by De Cock (2011). Kaggle scores submissions on the log scale, so an error of 10% costs about the same whether a house sells for $100,000 or $1,000,000."),
  body("This report follows the order in which we worked. Section 2 explains why we set aside a working first solution and rebuilt it. Sections 3 to 7 cover data exploration, our validation method, feature engineering, model comparison and blending. Section 8 covers what happened after we submitted, when the leaderboard did not agree with our CV. Every number in this report can be reproduced with our notebook (Appendix A)."),

  h1("2  Our first attempt and why we rebuilt it"),
  body("Our first submission, on 9 November, came from a pipeline we built with an AI coding assistant (Codex). It ran 27 experiments, blended ElasticNet with CatBoost, and scored 0.1212 on CV and 0.11745 on Kaggle, which was a decent result."),
  body("When we tried to explain it, though, we could not. The code wrapped everything in custom scikit-learn classes and many validation checks. We could see the final score, but we could not say why each feature was there or how much each choice had helped. Most of the marks for this assignment are for the pathway to the solution, so we decided to rebuild it in a way we could defend line by line."),
  body("The rebuilt version is one notebook in six parts. We gave ourselves two rules. Each step changes exactly one thing, and each step is scored with exactly the same five folds (scikit-learn's KFold with shuffling and random_state=42). With the folds fixed, a change in the score can only come from our own change. We used an AI assistant (Claude) during the rebuild as well, mostly as a tutor: it drafted each part and explained it, and we wrote our own study notes and only moved on once we could explain that part ourselves. Our separate statement on AI use gives more detail."),

  h1("3  Exploring the data"),
  body("SalePrice is strongly right-skewed, with a skewness of 1.88. The middle half of the houses sold for between $130,000 and $214,000, but a long tail reaches $755,000. After taking the log the skewness is 0.12, which is almost symmetric (Figure 1). This told us to predict log(SalePrice), which is also the scale Kaggle scores on."),
  ...figure("01_saleprice_distribution.png", 560, "Figure 1. SalePrice before and after taking the log."),
  body("Missing values come in two kinds, and the data description tells them apart. For many columns, NA means the house does not have the thing at all. PoolQC is missing for 1,453 of the 1,460 houses simply because almost no house has a pool. For other columns, such as LotFrontage (259 missing), NA means the value was not recorded. We treat the two kinds differently in Step 3. We also found 15 columns that have gaps only in the test set, for example MSZoning and GarageCars, so our preprocessing had to cope with gaps it never saw during training (Figure 2)."),
  ...figure("02_missing_values.png", 400, "Figure 2. The 20 columns with the most missing values in the training data. Orange means that NA stands for \"the house has none\"."),
  body("One small detail caught us out. By default, pandas reads the text \"None\" as a missing value, but in this data \"None\" is a real category: MasVnrType = None means the house has no masonry veneer. With the default settings, 872 houses looked as if their veneer type was unknown. We read the files with keep_default_na=False and na_values=[\"NA\", \"\"], which leaves only 8 that are truly unknown."),
  body("Living area is one of the strongest predictors, but two houses (Ids 524 and 1299) are over 4,000 sq ft and sold for less than $300,000 (Figure 3). Both were Partial sales, which the data description defines as homes that were not completed when last assessed, so their prices do not follow the usual pattern. Two other houses of that size sold for $745,000 and $755,000, which fits the pattern, so our outlier rule only covers large houses that sold below $300,000."),
  ...figure("03_outliers.png", 380, "Figure 3. Living area against sale price. The two circled houses are the outliers."),
  body("Among the numeric columns, OverallQual (0.79) and GrLivArea (0.71) have the highest correlation with SalePrice. Correlation only measures straight-line relationships and ignores the 43 text columns, so we used it as a quick map and did not drop any features because of it. Each finding in this section led to one of the feature steps in Section 5."),

  h1("4  How we validated our models"),
  body("Kaggle hides the test prices and limits how often we can submit, so we needed our own estimate of how well a model works on unseen houses. We used five-fold CV. The 1,460 houses are split into five groups of 292, and each group takes a turn as the validation set while a model is trained on the other four. Every house is predicted exactly once by a model that never saw it. The RMSE of these out-of-fold (OOF) predictions is our CV score. For the test set we average the predictions of the five fold models."),
  body("The main trap is data leakage. Filling gaps with the median, scaling, and listing the categories for one-hot encoding all learn something from the data. If we did them on all 1,460 houses first, the validation houses would influence training. For example, the median LotFrontage is 69 over all houses but 70 over the training part of the first fold. To avoid this, we put these steps and the model into a single scikit-learn Pipeline, which is refitted inside every fold. Changes that look at one house at a time, such as adding two areas together, cannot leak, so we apply those before CV."),
  body("Two baselines set the starting point. Predicting the average price for every house (Step 0) gives a CV RMSE of 0.408. A Ridge regression on the raw price with only the basic preprocessing (Step 1) gives 0.155, and this became our first submission (Kaggle 0.14188)."),

  h1("5  Feature engineering, one change at a time"),
  body("Table 1 lists the feature steps. Every row uses the same Ridge model (alpha = 10) and the same five folds, and changes one thing compared with the row above."),
  ...table(
    ["Step", "Change", "CV RMSE", "vs previous"],
    [
      ["1", "Ridge on the raw price (baseline)", "0.15534", ""],
      ["2", "Predict log(SalePrice)", "0.15206", "-0.00327"],
      ["3", "Label missing values by meaning; MSSubClass as a category", "0.15106", "-0.00100"],
      ["4", "New features: TotalSF, TotalBathrooms, HouseAge, RemodelAge, TotalPorchSF", "0.15121", "+0.00014"],
      ["5", "log1p of 20 skewed area columns", "0.13239", "-0.01882"],
      ["6", "Drop the 2 outliers from the training folds", "0.12766", "-0.00473"],
      ["6b", "Tune Ridge alpha (1 to 100)", "0.12766", "0"],
    ],
    [700, 5426, 1400, 1500],
    "Table 1. Feature engineering steps (Ridge, five-fold CV RMSE on log price)."
  ),
  body("Step 2 gave less than we expected. Predicting the log price lines up what the model learns with what Kaggle scores, but the features were still on their raw scale, and log price is not a straight line in raw area. The full benefit only appeared in Step 5, once the areas were logged too, so Steps 2 and 5 work as a pair."),
  body("In Step 3 we only replaced NA with a label such as NoGarage when another column confirmed it. A missing GarageType became NoGarage only if GarageArea was 0. If a house had a garage but its type was missing, we left it missing, so the Pipeline treated it as unknown. We also turned MSSubClass into text, because its values (20, 60, 190 and so on) are codes for house types rather than quantities. The gain was small (-0.0010). Looking back, this makes sense: the Pipeline was already filling text gaps with a \"Missing\" category, which a model can learn from almost as well. This step mostly made the data more accurate."),
  body("Step 4 added combined features such as TotalSF (living area plus basement area) and HouseAge (year sold minus year built). The CV score did not move (+0.00014). A linear model can already add two columns together by giving them the same weight, so TotalSF told Ridge nothing new. The features were still worth keeping. log(TotalSF) is not the sum of the separate logs, so they started to matter after Step 5, and tree models, which split on one column at a time, cannot build sums by themselves."),
  body("Step 5 was the biggest step. We took log1p, the log of 1 + x, so that zero areas stay at zero, of every area column whose skewness in the training data was above 0.75. That is 20 of our 21 area columns; only GarageArea (skewness 0.18) was left alone. The CV error fell by 0.0188, about 12%, and Figure 4 shows how much larger this drop is than the others. We chose the 20 columns by looking at the skewness of the whole training set, outside the CV loop. This uses the features only and never the price, so we believe its effect on the CV score is negligible."),
  ...figure("07_improvement_staircase.png", 560, "Figure 4. CV RMSE after each step of the pathway. The y-axis starts at 0.11, so the differences look larger than they are."),
  body("Step 6 removed the two outliers from the training data in each fold. They were still predicted and still counted in the score, because the test set could contain houses like them. CV improved by 0.0047. This step also gave us our most memorable bug. Our first version of the rule checked GrLivArea > 4000 inside the CV function, and Step 6 scored exactly the same as Step 5. Step 5 had already replaced GrLivArea with its log, whose maximum is about 8.6, so no house matched the rule. We fixed it by computing the outlier mask once from the raw data, and added a test that fails if the two steps ever score the same again."),
  body("Finally, we tuned Ridge's alpha over 1, 3, 10, 30 and 100. The scores formed a U shape, from 0.1303 at alpha = 1 down to 0.1277 at alpha = 10 and back up to 0.1298 at alpha = 100. Too little penalty overfits and too much underfits. Our first guess of 10 turned out to be the best value."),

  h1("6  Comparing models"),
  body("With the features fixed, we compared seven models on the same folds. All of them predict the log price and are trained without the two outliers (Table 2 and Figure 5)."),
  ...table(
    ["Model", "Family", "CV RMSE", "Time (s)"],
    [
      ["XGBoost", "Boosted trees", "0.12310", "4"],
      ["CatBoost", "Boosted trees", "0.12320", "86"],
      ["SVR (RBF kernel)", "Kernel", "0.12349", "5"],
      ["Lasso", "Linear", "0.12546", "2"],
      ["ElasticNet", "Linear", "0.12565", "3"],
      ["Ridge", "Linear", "0.12766", "0.4"],
      ["Random Forest", "Bagged trees", "0.13822", "12"],
    ],
    [2900, 2726, 1700, 1700],
    "Table 2. Single models (five-fold CV RMSE, and time for the full CV on our machine)."
  ),
  body("Lasso beat Ridge. Its penalty uses the absolute size of the coefficients, which pushes weak ones to exactly zero, and it kept only 107 of the 324 columns. Dropping columns that add little on top of the others seems to have reduced overfitting."),
  body("Random Forest was the weakest model, behind even Ridge. A forest averages many deep trees, each trained on a bootstrap sample. Averaging cancels random errors (variance) but not errors that every tree shares (bias). After Step 5, log price is close to a straight line in the main features, and a tree can only approximate a line with steps, so every tree makes the same kind of error. Boosting works differently. It adds small trees one after another, each fitted to the errors that are still left, so it can reduce that shared error. XGBoost (Chen and Guestrin, 2016) and CatBoost (Prokhorenkova et al., 2018), both boosting methods, came first and second. CatBoost handles text columns itself, replacing each category with an average price computed in an order that avoids leakage. It did as well as XGBoost without the log features or one-hot encoding, but it was about 20 times slower."),
  body("SVR needed the most tuning. Its two settings, C and gamma, partly cancel each other out, and the best pair kept landing on the edge of our grid. We widened the grid twice, then checked four pairs just outside the final corner (C = 1, gamma = 0.003), and all four scored worse. After this tuning SVR came third, at 0.1235."),
  ...figure("05_model_comparison.png", 430, "Figure 5. CV RMSE of the seven single models."),

  h1("7  Blending"),
  body("Averaging models only helps when they make different mistakes. To measure this, we correlated the models' out-of-fold residuals, that is, true minus predicted log price (Figure 6). Ridge, Lasso and ElasticNet correlate at 0.99 or more, so they make almost the same errors. The linear models and the boosted trees correlate at about 0.88, and XGBoost and CatBoost at 0.93, since they encode categories differently."),
  ...figure("06_oof_correlation.png", 360, "Figure 6. Correlation between the models' out-of-fold residuals. Darker cells mean more similar errors."),
  body("Before looking at any blend scores, we fixed a rule: take the best model from each family and keep both boosting models. That gave Lasso, XGBoost, CatBoost and SVR, averaged with equal weights. We did not tune the weights, because weights chosen on the same 1,460 OOF predictions would fit their noise and make the CV score look better than it really is. The blend scored 0.1191, better than the best single model (0.1231). Adding every model except Random Forest made it worse (0.1202), because three near-identical linear models then had three of the six votes."),

  h1("8  What the leaderboard taught us"),
  body("Table 3 puts our CV scores next to the Kaggle scores. Up to S4 the two agree and both go steadily down. After that they disagree. CatBoost (S5) beat Ridge (S4) on CV but was slightly worse on Kaggle, and the blend (S6), our best CV score, did worse on Kaggle than our first attempt (0.12037 against 0.11745)."),
  ...table(
    ["Submission", "What changed", "CV RMSE", "Kaggle"],
    [
      ["S1", "Ridge on the raw price", "0.15534", "0.14188"],
      ["S2", "Log target, missing values, new features", "0.15121", "0.13044"],
      ["S3", "Log of skewed areas", "0.13239", "0.12253"],
      ["S4", "Outliers dropped, alpha tuned", "0.12766", "0.12236"],
      ["S5", "CatBoost", "0.12320", "0.12310"],
      ["S6", "Four-model blend", "0.11906", "0.12037"],
      ["S7", "Outliers kept with a HugePartial flag", "0.11910", "0.11624"],
      ["First attempt", "ElasticNet + CatBoost (Codex pipeline)", "0.12117", "0.11745"],
    ],
    [1700, 4326, 1500, 1500],
    "Table 3. CV and Kaggle scores for every submission."
  ),
  body("Part of this is noise. When we scored the five folds separately, the same Ridge model ranged from 0.105 to 0.172, depending on which 292 houses were in the fold. A leaderboard score is one more sample of this kind, so a difference of 0.001 between two submissions means very little. We still wanted to know whether something real was going on, so we looked at where our CV error came from."),
  body("Two houses dominate it: the outliers 524 and 1299. The blend's log errors on them were -1.25 and -1.40, so it priced them at about 3.5 and 4 times their real value. Squared, these two errors make up about 17% of the total squared error, and without them our CV RMSE would be 0.1085 instead of 0.1191. Both houses also happen to fall into the same fold. Whenever one of them was being predicted, the other was missing from the training data too, so our CV could never test whether a model can learn from one of these houses and use it on the other."),
  body("The test set contains a third house of the same kind. Id 2550 has 5,095 sq ft of living area and the top quality rating of 10, and it was also a Partial sale. Steps 6 to 12 removed its two look-alikes from training, so our models had only learned that big houses are expensive, and they predicted between $565,000 and $763,000 for it (S4 to S6). Our first attempt predicted $491,000. Its CatBoost model had been trained with the outliers included and predicted $306,000 for this house, while the same model trained without them predicted $618,000. This house was the largest single difference between S6 and the first attempt."),
  body("Our fix, Step 13, keeps the two houses in training and adds a feature, HugePartial, equal to 1 for houses over 4,000 sq ft sold as Partial sales. A model can then learn a separate adjustment for this kind of sale without changing anything for normal houses. As in Step 6, the flag is computed from the raw area. We also cleaned an impossible value we found in the test set, a garage built in 2207 (Id 2593), by treating garage years later than the sale year plus one as missing."),
  body("Normal CV could not test this change, so we designed a small check of our own. We held out one of the two houses, trained on all the others, and predicted it, then did the same for the other house (Table 4). With our earlier method the blend was too high by a factor of 3.6 to 4.4. With Step 13 the error fell to a factor of 1.3 to 1.9."),
  ...table(
    ["Held-out house", "True price", "Earlier method (Steps 6 to 12)", "Step 13"],
    [
      ["Id 524", "$184,750", "$665,634 (log error -1.28)", "$242,533 (log error -0.27)"],
      ["Id 1299", "$160,000", "$708,551 (log error -1.49)", "$303,257 (log error -0.64)"],
    ],
    [1700, 1500, 2913, 2913],
    "Table 4. Leave-one-outlier-out check: the blend's prediction for each outlier, with (Step 13) or without (earlier method) the other outlier and the flag in the training data."
  ),
  body("We then refitted the same four models with the same settings and equal weights. As expected, the CV score hardly changed (0.11906 to 0.11910), because CV cannot see this effect. The prediction for Id 2550 fell from $643,000 to $260,000, while the other test houses changed by a median of only 0.25%. We submitted this once, as S7, and it scored 0.11624: our best score, and better than the first attempt. Since S6 and S7 differ almost only in that one house, the result suggests it sold for much less than our earlier predictions, like its two look-alikes."),
  body("We also tried several other ideas on the Step 13 features, all scored with the same CV, and kept none of them. Stacking with learned weights scored 0.1226 (checked with its own CV). The first attempt's recipe of Lasso plus CatBoost at 50/50 scored 0.1203. The first attempt's quality-score features made Lasso slightly worse (0.1256 to 0.1261), and averaging XGBoost over three random seeds did not change the blend (0.11910 to 0.11911)."),
  body("Step 13 was prompted by the leaderboard, so it is less clean than Steps 1 to 12, which were chosen on CV alone. To limit this, we fixed the change before submitting, checked it with the leave-one-out test, and submitted it only once instead of trying variants against the leaderboard."),

  h1("9  Conclusion and limitations"),
  body("From the Ridge baseline to the final blend, our CV error fell from 0.155 to 0.119 and our Kaggle score from 0.142 to 0.116. The largest single gain came from log-transforming the skewed areas. Boosted trees were the best single models, and averaging models that make different mistakes gave a further improvement at almost no cost."),
  body("Measuring one change at a time was the most useful habit we picked up. It showed us which ideas mattered, and some results surprised us, such as new features that did nothing for a linear model and a Random Forest that lost to Ridge. Our CV also had a blind spot that only the leaderboard exposed, and looking at individual errors helped us more than trying more models."),
  body("Our work has clear limits. All our choices (features, settings and blend members) were made with the same CV, so our CV scores are a little optimistic. Step 13 rests on only two training examples. Many of the leaderboard differences between our submissions are smaller than the noise between folds. If we did this again, we would submit after every single step, since S2 bundled three changes and we could not split its leaderboard gain between them. We would also make sure that rare kinds of houses are spread across different folds."),

  new Paragraph({ children: [new PageBreak()] }),
  h1("Appendix A  Notebook and code"),
  body("Kaggle notebook: [link]. The notebook rebuilds every step in this report and runs in a few minutes on Kaggle; most of the time is spent on CatBoost. It is also in our repository, together with our study notes: [link]."),

  h1("Appendix B  Individual answers"),
  body("[Each member writes their own answers: main role in the competition; main difficulties for the team and for themselves; what they learned.]"),

  h1("References"),
  reference("Chen, T. and Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785-794."),
  reference("De Cock, D. (2011). Ames, Iowa: Alternative to the Boston housing data as an end of semester regression project. Journal of Statistics Education, 19(3)."),
  reference("Kaggle. House Prices: Advanced Regression Techniques. https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques"),
  reference("Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, pp. 2825-2830."),
  reference("Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. and Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In Advances in Neural Information Processing Systems 31."),
];

const doc = new Document({
  creator: "[team name]",
  title: "House Prices Kaggle report",
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true },
        paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, italics: true },
        paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 1 },
      },
    ],
  },
  sections: [{
    properties: {
      page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  const out = path.join(__dirname, "House_Prices_Report_draft.docx");
  fs.writeFileSync(out, buffer);
  console.log("wrote", out);
});
