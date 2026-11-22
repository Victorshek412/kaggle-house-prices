"""Regression checks for errors that would invalidate a competition experiment."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import house_prices as hp


class HousingWorkflowTests(unittest.TestCase):
    def test_csv_preserves_none_category_but_recognizes_na(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("Id,MasVnrType,MasVnrArea\n1,None,0\n2,NA,198\n")
            data = hp.read_competition_csv(path)
        self.assertEqual(data.loc[0, "MasVnrType"], "None")
        self.assertTrue(pd.isna(data.loc[1, "MasVnrType"]))

    def test_semantic_missingness_uses_evidence_and_never_changes_input(self):
        data = pd.DataFrame({"Id": [1, 2], "SalePrice": [100000, 200000],
                             "TotalBsmtSF": [0, 900], "BsmtQual": [np.nan, np.nan],
                             "PoolArea": [0, 80], "PoolQC": [np.nan, np.nan],
                             "MasVnrType": ["None", np.nan], "MSSubClass": [20, 150]})
        before = data.copy(deep=True)
        result = hp.FeatureBuilder(groups=("semantic",)).fit_transform(data)
        self.assertEqual(result["BsmtQual"].tolist(), ["NoBasement", "Unknown"])
        self.assertEqual(result["PoolQC"].tolist(), ["NoPool", "Unknown"])
        self.assertEqual(result["MasVnrType"].tolist(), ["None", "Unknown"])
        self.assertNotIn("Id", result)
        self.assertNotIn("SalePrice", result)
        self.assertEqual(result["MSSubClass"].tolist(), ["20", "150"])
        pd.testing.assert_frame_equal(data, before)

    def test_features_do_not_double_count_living_area_or_half_baths(self):
        data = pd.DataFrame({"GrLivArea": [1500.], "1stFlrSF": [1000.],
                             "2ndFlrSF": [500.], "TotalBsmtSF": [800.],
                             "FullBath": [2.], "BsmtFullBath": [1.],
                             "HalfBath": [1.], "BsmtHalfBath": [1.],
                             "YearBuilt": [2000], "YearRemodAdd": [2008],
                             "YrSold": [2007], "OverallQual": [7]})
        result = hp.FeatureBuilder(groups=("size", "age", "quality")).fit_transform(data)
        self.assertEqual(result.loc[0, "TotalSF"], 2300.)
        self.assertEqual(result.loc[0, "TotalBathrooms"], 4.)
        self.assertEqual(result.loc[0, "HouseAge"], 7.)
        self.assertEqual(result.loc[0, "RemodelAge"], 0.)
        self.assertEqual(result.loc[0, "RemodelAfterSale"], 1)
        self.assertEqual(result.loc[0, "QualityLivingArea"], 10500.)

    def test_pipeline_does_not_learn_validation_median_or_categories(self):
        train = pd.DataFrame({"Id": [1, 2, 3, 4], "LotArea": [10., 20., np.nan, 30.],
                              "Neighborhood": ["A", "B", "A", "B"]})
        valid = pd.DataFrame({"Id": [5, 6], "LotArea": [100000., np.nan],
                              "Neighborhood": ["UNSEEN", "UNSEEN"]})
        model = hp.make_model("ridge", groups=(), parameters={"alpha": 1.0})
        model.fit(train, np.log([100000, 120000, 110000, 130000]))
        preprocess = model.named_steps["preprocess"]
        imputer = preprocess.named_transformers_["numeric"].named_steps["imputer"]
        np.testing.assert_allclose(imputer.statistics_, [20.])
        pred = model.predict(valid)
        self.assertTrue(np.isfinite(pred).all())
        np.testing.assert_allclose(imputer.statistics_, [20.])
        encoder = preprocess.named_transformers_["categorical"].named_steps["encoder"]
        self.assertNotIn("UNSEEN", encoder.categories_[0])

    def test_outlier_exclusion_only_affects_training_not_validation_coverage(self):
        data = pd.DataFrame({"Id": range(1, 21), "GrLivArea": [1000.] * 19 + [5000.]})
        target = pd.Series([100000.] * 20)
        folds = hp.make_folds(len(data), seed=42)
        result = hp.evaluate_model(hp.make_model("ridge", groups=()), data, target,
                                   data.iloc[:2], folds, exclude_outliers=True)
        self.assertEqual(len(result["oof_log_predictions"]), 20)
        self.assertTrue(np.isfinite(result["oof_log_predictions"]).all())
        self.assertEqual(sum(row["validation_rows"] for row in result["folds"]), 20)
        self.assertEqual(sum(row["removed_training_rows"] for row in result["folds"]), 4)

    def test_submission_aligns_ids_and_rejects_invalid_predictions(self):
        sample = pd.DataFrame({"Id": [1461, 1462], "SalePrice": [0., 0.]})
        result = hp.make_submission([1462, 1461], np.log([200000., 100000.]), sample)
        self.assertEqual(result.columns.tolist(), ["Id", "SalePrice"])
        self.assertEqual(result.Id.tolist(), [1461, 1462])
        np.testing.assert_allclose(result.SalePrice, [100000., 200000.])
        for ids, predictions in [([1461, 1461], [11., 12.]),
                                 ([1461, 1463], [11., 12.]),
                                 ([1461, 1462], [np.nan, 12.]),
                                 ([1461, 1462], [1000., 12.])]:
            with self.subTest(ids=ids, predictions=predictions):
                with self.assertRaises(ValueError):
                    hp.make_submission(ids, predictions, sample)


if __name__ == "__main__":
    unittest.main()
