import argparse
import json
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# name -> (estimator, param grid)  — same models/grids as the notebook
MODELS = {
    "linear_regression": (
        LinearRegression(),
        {"fit_intercept": [True, False]},
    ),
    "decision_tree": (
        DecisionTreeRegressor(random_state=42),
        {"max_depth": [5, 10, 20], "min_samples_split": [2, 5]},
    ),
    "random_forest": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {"n_estimators": [50, 100], "max_depth": [10, 20]},
    ),
    "gradient_boosting": (
        GradientBoostingRegressor(random_state=42),
        {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS.keys()))
    ap.add_argument("--datadir", default="data/processed")
    ap.add_argument("--outdir", default="models")
    args = ap.parse_args()

    train = pd.read_csv(os.path.join(args.datadir, "train_prepared.csv"))
    test = pd.read_csv(os.path.join(args.datadir, "test_prepared.csv"))
    X_train, y_train = train.drop(columns=["sellingprice"]), train["sellingprice"]
    X_test, y_test = test.drop(columns=["sellingprice"]), test["sellingprice"]

    estimator, grid = MODELS[args.model]
    search = GridSearchCV(estimator, grid, cv=3, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train)

    pred = search.predict(X_test)
    metrics = {
        "model": args.model,
        "best_params": search.best_params_,
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }
    print(json.dumps(metrics, indent=2))

    os.makedirs(args.outdir, exist_ok=True)
    joblib.dump(search.best_estimator_, os.path.join(args.outdir, f"{args.model}.joblib"))
    with open(os.path.join(args.outdir, f"{args.model}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
