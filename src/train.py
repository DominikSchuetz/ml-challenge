import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from prepare_data import read_and_prepare_train

SEED = 42

def main(args):
    # 1) Daten laden & aggregieren
    train_user = read_and_prepare_train(args.train_path)

    # Features/Target trennen
    y = train_user["gender"].values
    X = train_user.drop(columns=["gender"])

    # Spalten definieren
    text_col = "doc"
    numeric_cols = [c for c in X.columns if c not in ["user_id", text_col]]

    # 2) Pipeline aufbauen
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000
    )

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False))  # mit sparse vereinbar
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("tfidf", tfidf, text_col),
            ("num", num_pipe, numeric_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0
    )

    clf = LogisticRegression(
        max_iter=2000,
        solver="saga",
        n_jobs=-1,
        class_weight="balanced",
        random_state=SEED,
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("clf", clf)
    ])

    # 3) Cross-Validation + Hyperparam-Tuning
    param_grid = {                                              
        "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0],                    
        "preprocess__tfidf__max_features": [50_000, 100_000],   
    }
    # typische Parameter für lineare Modelle mit hoher Dimension
    # nicht zu feinmaschig → beschleunigt die GridSearch deutlich
    # gut geeignet, um Under- vs. Overfitting zu erkennen


    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    scorer = "roc_auc"  # auch "f1_macro" möglich, aber AUC ist hier sinnvoller
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

    # 4) Auf vollständigen Trainingsdaten final trainiertes Modell speichern
    best_model = grid.best_estimator_
    joblib.dump(best_model, args.model_out)
    print(f"Saved model to {args.model_out}")

    # 5) Evaluation (Train-CV-Report für Transparenz)
    oof_pred_proba = np.zeros(len(X))
    oof_pred_label = np.empty(len(X), dtype=object)

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X, y), 1):
        print(f"CV fold {fold}")
        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y[trn_idx], y[val_idx]

        model = grid.best_estimator_
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_val)
        # Index der positiven Klasse (z.B. "male" – hängt von y ab)
        pos_idx = list(model.classes_).index(model.classes_[1])  
        oof_pred_proba[val_idx] = proba[:, pos_idx]
        oof_pred_label[val_idx] = model.predict(X_val)

    try:
        auc = roc_auc_score(y_true=(y == model.classes_[1]).astype(int), y_score=oof_pred_proba)
        print(f"OOF ROC-AUC: {auc:.4f}")
    except Exception:
        print("ROC-AUC konnte nicht berechnet werden (Labels evtl. nicht binär).")

    print("\nOOF classification report:")
    print(classification_report(y, oof_pred_label))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--model_out", type=str, default="models/model.joblib")
    args = parser.parse_args()
    main(args)
