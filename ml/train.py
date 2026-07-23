"""Train the two vehicle-maintenance classifiers.

Loads the labelled sensor dataset, trains a set of candidate models inside a
StandardScaler pipeline, selects the best per track by 5-fold cross-validated
weighted F1, and saves the chosen pipelines to the models directory. The saved
artifact is a full Pipeline(StandardScaler, clf), so inference does not need to
scale features separately.

Usage:
    python ml/train.py                          # data/Vehicle_Sensor_Dataset.xlsx -> models/
    python ml/train.py --data path.xlsx --out models/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from predictivecare import config
from predictivecare.features import (
    TRACK1_FEATURES, TRACK1_TARGET, TRACK2_FEATURES, TRACK2_TARGET,
)

RANDOM_STATE = 42
TEST_SIZE = 0.40
CV_FOLDS = 5

SHEET_TRACK1 = "Track1_Technician_30Day"
SHEET_TRACK2 = "Track2_Owner_12Hr"


def get_models() -> dict:
    """The candidate classifiers. Best-per-track is chosen by CV F1."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }


def build_pipe(clf) -> Pipeline:
    """Wrap a classifier in a StandardScaler pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train_track(df: pd.DataFrame, features: list[str], target: str,
                cv: StratifiedKFold) -> tuple[Pipeline, str]:
    """Train all candidates for one track and return the best pipeline + name."""
    X = df[features].values
    y = df[target].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )

    best_name, best_pipe, best_cv = "", None, -1.0
    print(f"{'Model':<25}{'Test Acc':>10}{'Test F1':>10}{'CV F1':>10}")
    print("-" * 55)
    for name, clf in get_models().items():
        pipe = build_pipe(clf).fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        f1_test = f1_score(y_te, y_pred, average="weighted")
        cv_mean = cross_val_score(
            build_pipe(clf), X, y, cv=cv, scoring="f1_weighted", n_jobs=-1,
        ).mean()
        acc = float((y_pred == y_te).mean())
        print(f"{name:<25}{acc:>10.4f}{f1_test:>10.4f}{cv_mean:>10.4f}")
        if cv_mean > best_cv:
            best_name, best_pipe, best_cv = name, pipe, cv_mean

    print(f"\nBest: {best_name}  (CV F1 = {best_cv:.4f})\n")
    print(classification_report(y_te, best_pipe.predict(X_te), zero_division=0))
    return best_pipe, best_name


def main(argv=None) -> None:
    default_data = Path(config.DATASET_PATH).parent / "Vehicle_Sensor_Dataset.xlsx"
    ap = argparse.ArgumentParser(description="Train the vehicle-maintenance classifiers.")
    ap.add_argument("--data", default=str(default_data), help="labelled training .xlsx")
    ap.add_argument("--out", default=str(config.MODELS_DIR), help="output directory for .pkl models")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    xl = pd.ExcelFile(args.data)
    df1 = xl.parse(SHEET_TRACK1)
    df2 = xl.parse(SHEET_TRACK2)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("\n=== Track 1: Fault Classification ===")
    pipe1, name1 = train_track(df1, TRACK1_FEATURES, TRACK1_TARGET, cv)
    joblib.dump(pipe1, out / "track1_fault_classifier.pkl")

    print("\n=== Track 2: Risk Detection ===")
    pipe2, name2 = train_track(df2, TRACK2_FEATURES, TRACK2_TARGET, cv)
    joblib.dump(pipe2, out / "track2_risk_classifier.pkl")

    print(f"\nSaved models to {out}:")
    print(f"  track1_fault_classifier.pkl  ({name1})")
    print(f"  track2_risk_classifier.pkl   ({name2})")


if __name__ == "__main__":
    main()
