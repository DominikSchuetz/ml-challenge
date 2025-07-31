import argparse
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from prepare_data import read_and_prepare_train

SEED = 42

def main(args):
    # 1) Daten laden & Feature-Matrix erstellen
    train_user = read_and_prepare_train(args.train_path)
    y = train_user["gender"].values
    X = train_user.drop(columns=["gender"])

    text_col = "doc"
    numeric_cols = [c for c in X.columns if c not in ["user_id", text_col]]

    # 2) Pipeline für Textdaten (TF-IDF + SVD)
    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000)),
        ("svd", TruncatedSVD(n_components=300, random_state=SEED)),
        ("scaler", StandardScaler())  # SVD -> dense -> StandardScaler mit mean ok
    ])

    # 3) Pipeline für numerische Features
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 4) Spaltenweiser ColumnTransformer
    preprocess = ColumnTransformer(
        transformers=[
            ("text", text_pipe, text_col),
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0  # wollen ein dichtes Array für LightGBM
    )

    # 5) LightGBM-Klassifikator definieren
    clf = LGBMClassifier(
        objective="binary" if len(np.unique(y)) == 2 else "multiclass",
        n_estimators=1000,
        random_state=SEED,
        n_jobs=-1
    )

    # 6) Gesamte Pipeline zusammenbauen
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("clf", clf)
    ])

    # 7) Hyperparameter-Raster für GridSearch
    param_grid = {
        "clf__learning_rate": [0.05, 0.1],
        "clf__num_leaves": [31, 63, 127],
        "clf__min_child_samples": [10, 50, 100],
        "clf__subsample": [0.8, 1.0],
        "preprocess__text__tfidf__max_features": [30000, 50000],
        "preprocess__text__svd__n_components": [100, 200],
    }

    # 8) Cross-Validation Setup
    cv = StratifiedKFold(n_splits=3, random_state=SEED, shuffle=True)
    scorer = "roc_auc" if len(np.unique(y)) == 2 else "f1_macro"

    # 9) GridSearchCV starten
    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)
    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV {scorer}: {grid.best_score_:.4f}")

    # 10) Bestes Modell speichern
    best_model = grid.best_estimator_
    joblib.dump(best_model, args.model_out)
    print(f"Saved model to {args.model_out}")

    # 11) Evaluation per Cross-Validation (OOF)
    # OOF-Eval (optional: schnelle Grob-Einschätzung)
    print("\n(Re-)Fitting best model on full train and reporting CV again (quick):")
    oof_probs = np.zeros(len(X))
    oof_preds = np.empty(len(X), dtype=object)

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y[trn_idx], y[val_idx]

        model = grid.best_estimator_
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_val)
        if len(model.named_steps["clf"].classes_) == 2:
            positive = list(model.named_steps["clf"].classes_).index(model.named_steps["clf"].classes_[1])
            oof_probs[val_idx] = proba[:, positive]
        oof_preds[val_idx] = model.predict(X_val)

    # 12) OOF Evaluation & Report
    if len(np.unique(y)) == 2:
        y_bin = (y == model.named_steps["clf"].classes_[1]).astype(int)
        print(f"OOF ROC-AUC: {roc_auc_score(y_bin, oof_probs):.4f}")

    print("\nOOF Classification report:")
    print(classification_report(y, oof_preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--model_out", type=str, default="models/lgbm_svd_model.joblib")
    args = parser.parse_args()
    main(args)
