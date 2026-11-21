"""Build the standalone project notebook; optionally save rendered results."""

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "House_Prices_Project.ipynb"
CACHE_FILES = ("experiments.csv", "feature_ablation.csv", "stability.csv", "session_summary.json", "provenance.json")


def markdown(source):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source, *, hidden=False):
    metadata = {"jupyter": {"source_hidden": True}} if hidden else {}
    return nbf.v4.new_code_cell(dedent(source).strip(), metadata=metadata)


def module_cell(name, variable):
    source = (ROOT / f"{name}.py").read_text(encoding="utf-8")
    # Keep normal source line breaks so students can inspect the embedded code.
    quote = "'''" if "'''" not in source else '"""'
    literal = f"r{quote}\n{source}{quote}" if quote not in source else repr(source)
    return code(f"{variable} = _project_module({name!r}, {literal})", hidden=True)


def build():
    return nbf.v4.new_notebook(
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python", "file_extension": ".py"},
        },
        cells=[
            markdown(r"""
                # House Prices: from a baseline to a submission

                Predict the final sale price of homes in Ames, Iowa. This notebook records
                our first experiment session: the data audit, validation design, feature
                comparisons, model comparisons, and submission candidates. The tables below
                report **local validation**; they do not establish a Kaggle score or rank.

                **Read:** use the saved outputs to review the work. **Run:** keep this notebook
                beside the local competition data and `results/`, or attach the official
                competition data to a Kaggle notebook. The source is embedded below; you do
                not need to upload the two Python modules. If the saved tables are absent,
                Run All trains the models and writes `results/` and `submissions/`.
            """),
            code("""
                from pathlib import Path
                import json
                import hashlib
                import sys
                from types import ModuleType

                import numpy as np
                import pandas as pd
                from IPython.display import Image, Markdown, display

                RUN_EXPERIMENTS = False  # True forces a fresh run even if results exist.
                INCLUDE_OUTLIER_COMPARISON = True
                DATA_DIR = None  # Or set Path('/path/to/folder/with/train.csv').

                COMPETITION = 'house-prices-advanced-regression-techniques'
                kaggle_working = Path('/kaggle/working')
                OUTPUT_DIR = kaggle_working if kaggle_working.is_dir() else Path.cwd()
                RESULTS = OUTPUT_DIR / 'results'

                candidates = [Path.cwd() / COMPETITION, Path.cwd(),
                              Path('/kaggle/input') / COMPETITION,
                              Path('/kaggle/input/competitions') / COMPETITION]
                if DATA_DIR is None:
                    DATA_DIR = next((path for path in candidates if (path / 'train.csv').is_file()), None)
                else:
                    DATA_DIR = Path(DATA_DIR)

                def _project_module(name, source):
                    # Register an in-memory module, so imports and dataclasses work normally.
                    source = source.lstrip('\\n')
                    module = ModuleType(name)
                    module.__file__ = str(Path.cwd() / (name + '.py'))
                    module.__source__ = source
                    sys.modules[name] = module
                    exec(compile(source, module.__file__, 'exec'), module.__dict__)
                    return module
            """),
            markdown("""
                ## 1. Reusable code

                Expand the next two code cells to inspect the feature builder, preprocessing
                pipelines, cross-validation loop and experiment runner. They contain the same
                source as the local scripts and run directly in the notebook kernel.
            """),
            module_cell("house_prices", "hp"),
            module_cell("run_experiments", "runner"),
            markdown("""
                ## 2. Data audit

                `train.csv` contains the measured sale prices; `test.csv` has the same type
                of house information without the prices. We keep `Id` to match rows for export
                and exclude it from the predictors.

                An empty or `NA` value does not have one universal meaning. For fields such
                as `PoolQC` and `Alley`, the data dictionary gives a structural reason (no
                pool or alley access). Other fields can be unknown. The literal string `None`
                is a recorded category, for example in `MasVnrType`; we preserve it separately
                from missing values. The audit below counts raw missing values before preprocessing.
            """),
            code("""
                if DATA_DIR is None:
                    print('Attach the competition data or set DATA_DIR to rerun the audit.')
                else:
                    train, test, sample = hp.load_data(DATA_DIR)
                    print(f'Training rows: {len(train):,} | Test rows: {len(test):,}')
                    print(f'Training examples with literal None for MasVnrType: '
                          f'{train["MasVnrType"].eq("None").sum():,}')
                    missing = pd.DataFrame({'Train': train.isna().sum(), 'Test': test.isna().sum()})
                    display(missing.assign(total=missing.sum(axis=1)).sort_values('total', ascending=False)
                                   .head(12).drop(columns='total'))
            """),
            markdown(r"""
                ## 3. How we score a model

                Kaggle evaluates the root mean squared error between the logarithms of the
                observed and predicted prices. We train on the natural log of `SalePrice`.
                On the log scale, predicting \$200,000 for a \$100,000 home incurs the same
                error magnitude as predicting \$400,000 for a \$200,000 home: both double
                the real price.

                $$\mathrm{log\mbox{-}RMSE} =
                \sqrt{\frac{1}{n}\sum_i\left(\log(y_i)-\widehat{\log(y_i)}\right)^2}$$

                We shuffle the **1,460 labeled rows** into five folds with seed 42. Each fold
                trains on four parts and predicts the fifth. Learned preprocessing, such as
                imputing numeric values, scaling and fitting the category encoder, uses only
                that fold's training rows. The primary score pools the 1,460 held-out predictions.
                The fold mean and standard deviation summarize how much scores vary across folds.

                The optional outlier comparison uses the same folds. It excludes rows with
                `GrLivArea > 4000` and `SalePrice < 300000` **only during training**. Every
                validation row still counts, including unusual homes. The rule uses known
                labels, so it is never applied to the competition test set.

                To build a submission, we average the five fitted models' test predictions
                on the log scale, then use `exp` to return to dollar prices. The fixed model
                blend also averages log predictions before converting to dollars.
            """),
            markdown("""
                ## 4. Run or load the experiments

                The constant baseline predicts the same log price for every home. Ridge
                gives us a simple reference with the original predictors, then we vary the
                feature groups while keeping the folds and model settings fixed. Elastic Net
                gives a second linear model; SVR adds a kernel-based comparison; CatBoost and XGBoost let us compare tree models
                that can learn nonlinear relationships and interactions. The comparison uses
                a small set of preset configurations.
            """),
            code(f"""
                cache_files = {CACHE_FILES!r}
                source_hashes = {{name + '.py': hashlib.sha256(module.__source__.encode()).hexdigest()
                                 for name, module in [('house_prices', hp), ('run_experiments', runner)]}}
                if RUN_EXPERIMENTS or not all((RESULTS / name).is_file() for name in cache_files):
                    if DATA_DIR is None:
                        raise FileNotFoundError('Attach the official competition data or set DATA_DIR.')
                    runner.run_session(data_dir=DATA_DIR, output_dir=OUTPUT_DIR,
                                       include_outliers=INCLUDE_OUTLIER_COMPARISON)
                    provenance_path = RESULTS / 'provenance.json'
                    provenance = json.loads(provenance_path.read_text())
                    provenance['source_sha256'] = source_hashes
                    provenance['source_origin'] = 'Executed embedded notebook modules'
                    provenance_path.write_text(json.dumps(provenance, indent=2) + '\\n')
                else:
                    provenance = json.loads((RESULTS / 'provenance.json').read_text())
                    if provenance.get('source_sha256') != source_hashes:
                        raise ValueError('Saved results use different source code. Set RUN_EXPERIMENTS = True to regenerate.')
                    if DATA_DIR is not None:
                        for name, expected in provenance['data_sha256'].items():
                            if hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() != expected:
                                raise ValueError('Saved results use different data. Set RUN_EXPERIMENTS = True to regenerate.')
                    print('Showing saved local experiment results.')
            """),
            markdown("""
                ## 5. What changed the score?

                Compare `cv_rmse` across rows; lower is better. Use the feature ablation
                table to compare engineered groups with Ridge held fixed, so a change is not
                confounded with changing the model family. A group can help a linear model
                even if a tree already learns the same relationship from the original columns.
            """),
            code("""
                experiments = pd.read_csv(RESULTS / 'experiments.csv', keep_default_na=False)
                development = experiments[experiments['seed'].eq(42)]
                display(development.sort_values('cv_rmse').reset_index(drop=True))
                print('Only seed-42 development runs are ranked here; the second seed is shown separately below.')
            """),
            code("""
                ablation = pd.read_csv(RESULTS / 'feature_ablation.csv')
                display(ablation)
            """),
            code("""
                stability = pd.read_csv(RESULTS / 'stability.csv')
                display(stability)
                print('Second-seed results check split sensitivity; they are not an untouched test set.')
            """),
            markdown("""
                ## 6. Check the fit

                Each out-of-fold prediction comes from a model that did not train on that
                home. Use the plots to look for large misses and systematic under- or
                overprediction. A few unusual homes may affect RMSE more than their count
                suggests because it squares errors. A second shuffled split is a stability
                check on these same homes. Our local comparisons informed model choice, so
                the best displayed score may be optimistic.
            """),
            code("""
                for figure in sorted((RESULTS / 'figures').glob('*.png')):
                    display(Markdown('**' + figure.stem.replace('_', ' ').capitalize() + '**'))
                    display(Image(filename=str(figure)))
            """),
            markdown("""
                ## 7. Submission candidates and next decisions

                The runner checks that each CSV has `Id,SalePrice`, uses the sample
                submission's exact ID order, and contains a positive finite price for every
                test home. Review the development and stability evidence before choosing a
                candidate to upload. Record an actual Kaggle result separately after a manual
                upload. The top 5% target remains an objective until the leaderboard shows it.
            """),
            code("""
                summary = json.loads((RESULTS / 'session_summary.json').read_text())
                display(summary)
                for candidate in sorted((OUTPUT_DIR / 'submissions').glob('*.csv')):
                    print(candidate.name)
            """),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Save notebook outputs using existing results.")
    args = parser.parse_args()
    notebook = build()
    if args.execute:
        from nbclient import NotebookClient

        if not all((ROOT / "results" / name).is_file() for name in CACHE_FILES):
            parser.error("Run run_experiments.py before using --execute.")
        NotebookClient(notebook, timeout=120, resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.validate(notebook)
    nbf.write(notebook, NOTEBOOK)
    print(f"Wrote {NOTEBOOK.name}")


if __name__ == "__main__":
    main()
