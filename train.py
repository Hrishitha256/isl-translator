"""
train.py - train the sign classifier from EVERY CSV in data/.

Run:  python train.py

- 1 person's data  -> random 80/20 split (optimistic score, pipeline test only)
- 2+ people's data -> leave-one-person-out: train on everyone else, test on
                      the person left out (the honest score for a new signer)

The best model is retrained on ALL data and saved to models/model.pkl
"""

import glob
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = "data"
MODEL_PATH = os.path.join("models", "model.pkl")


def load_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        raise SystemExit("No CSV files found in data/. Record some samples first.")
    X_parts, y_parts, p_parts = [], [], []
    for f in files:
        df = pd.read_csv(f, dtype={"label": str})   # keep "1" as text, not a number
        if len(df) == 0:
            print(f"Skipping empty file: {f}")
            continue
        name = os.path.splitext(os.path.basename(f))[0]
        X_parts.append(df.drop(columns=["label"]).to_numpy(dtype=np.float32))
        y_parts.append(df["label"].to_numpy())
        p_parts.append(np.full(len(df), name))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    person = np.concatenate(p_parts)
    return X, y, person


def make_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    }


def evaluate(name, model, X, y, person):
    """Return (accuracy, true labels, predicted labels) using the honest split."""
    people = np.unique(person)
    if len(people) >= 2:
        y_true_all, y_pred_all = [], []
        print(f"\n{name}: leave-one-person-out")
        for p in people:
            test = person == p
            model.fit(X[~test], y[~test])
            pred = model.predict(X[test])
            acc = accuracy_score(y[test], pred)
            print(f"   test on {p:<12} accuracy = {acc:.3f}")
            y_true_all.extend(y[test])
            y_pred_all.extend(pred)
        acc = accuracy_score(y_true_all, y_pred_all)
        print(f"   overall accuracy = {acc:.3f}")
        return acc, np.array(y_true_all), np.array(y_pred_all)

    print(f"\n{name}: random 80/20 split (only one person's data - optimistic!)")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    acc = accuracy_score(y_te, pred)
    print(f"   accuracy = {acc:.3f}")
    return acc, y_te, pred


def main():
    X, y, person = load_data()
    print(f"Loaded {len(y)} samples from {len(np.unique(person))} person(s): "
          f"{', '.join(np.unique(person))}")
    print("Samples per sign:")
    print(pd.Series(y).value_counts().sort_index().to_string())

    results = {}
    for name, model in make_models().items():
        acc, y_true, y_pred = evaluate(name, model, X, y, person)
        results[name] = (acc, y_true, y_pred)

    best = max(results, key=lambda k: results[k][0])
    acc, y_true, y_pred = results[best]
    print(f"\nBest model: {best} (accuracy {acc:.3f})")

    # confusion matrix: rows = true sign, columns = predicted sign
    cm = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="predicted"))
    print("\nConfusion matrix (rows = true, columns = predicted):")
    print(cm.to_string())

    # list the most confused pairs
    confused = [(t, p, int(cm.loc[t, p])) for t in cm.index for p in cm.columns
                if t != p and cm.loc[t, p] > 0]
    confused.sort(key=lambda r: -r[2])
    if confused:
        print("\nMost confused pairs (true -> predicted : count):")
        for t, p, c in confused[:8]:
            print(f"   {t} -> {p} : {c}")
    else:
        print("\nNo mistakes on the test data.")

    # retrain the best model on ALL data and save it
    final = make_models()[best]
    final.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    print(f"\nSaved {best} trained on all {len(y)} samples to {MODEL_PATH}")
    print("Signs it knows:", list(final.classes_))


if __name__ == "__main__":
    main()