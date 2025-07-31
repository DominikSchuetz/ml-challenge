import argparse
import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tqdm import tqdm

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier

from prepare_data import read_and_prepare_train, get_user_path_sequences

SEED = 42

def average_embedding(paths, w2v_model, embed_dim):
    vecs = []
    for p in paths:
        if p in w2v_model.wv:
            vecs.append(w2v_model.wv[p])
    if len(vecs) == 0:
        return np.zeros(embed_dim, dtype=np.float32)
    return np.mean(vecs, axis=0)

def main(args):
    # 1) Aggregierte Features
    train_user = read_and_prepare_train(args.train_path)
    y = train_user["gender"].values
    X_num = train_user.drop(columns=["gender", "user_id", "doc"])  # numerische Features

    # 2) Sequenzen für Word2Vec
    seq_df = get_user_path_sequences(args.train_path, has_gender=True)
    sequences = seq_df["paths"].tolist()

    # 3) Word2Vec trainieren
    print("Training Word2Vec ...")
    w2v = Word2Vec(
        sentences=sequences,
        vector_size=args.embed_dim,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        epochs=10,
        seed=SEED
    )
    w2v.save(args.w2v_out)
    print(f"Saved Word2Vec model to {args.w2v_out}")

    # 4) User-Embeddings bauen
    print("Building user embeddings ...")
    user_embeds = []
    for paths in tqdm(seq_df["paths"], total=len(seq_df)):
        user_embeds.append(average_embedding(paths, w2v, args.embed_dim))
    user_embeds = np.vstack(user_embeds)

    # Reihenfolge sicherstellen
    assert (seq_df["user_id"].values == train_user["user_id"].values).all(), \
        "user_id order mismatch between seq_df and train_user. Sort both by user_id first."

    # 5) Numerische Features skalieren & zusammenführen
    num_imputer = SimpleImputer(strategy="median")
    num_scaler = StandardScaler()

    X_num_imp = num_imputer.fit_transform(X_num)
    X_num_scaled = num_scaler.fit_transform(X_num_imp)

    X_full = np.hstack([user_embeds, X_num_scaled])

    # 6) LightGBM + CV + GridSearch
    clf = LGBMClassifier(
        objective="binary" if len(np.unique(y)) == 2 else "multiclass",
        n_estimators=1000,
        random_state=SEED,
        n_jobs=-1
    )

    param_grid = {
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63, 127],
        "min_child_samples": [10, 50, 100],
        "subsample": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    best_score = -np.inf
    best_params = None
    best_model = None

    for lr in param_grid["learning_rate"]:
        for nl in param_grid["num_leaves"]:
            for mcs in param_grid["min_child_samples"]:
                for subs in param_grid["subsample"]:
                    scores = []
                    for trn_idx, val_idx in cv.split(X_full, y):
                        X_tr, X_val = X_full[trn_idx], X_full[val_idx]
                        y_tr, y_val = y[trn_idx], y[val_idx]

                        model = LGBMClassifier(
                            objective="binary" if len(np.unique(y)) == 2 else "multiclass",
                            n_estimators=1000,
                            random_state=SEED,
                            n_jobs=-1,
                            learning_rate=lr,
                            num_leaves=nl,
                            min_child_samples=mcs,
                            subsample=subs
                        )
                        model.fit(X_tr, y_tr)
                        if len(np.unique(y)) == 2:
                            proba = model.predict_proba(X_val)[:, 1]
                            score = roc_auc_score((y_val == model.classes_[1]).astype(int), proba)
                        else:
                            preds = model.predict(X_val)
                            score = f1_score(y_val, preds, average="macro")
                        scores.append(score)

                    mean_score = np.mean(scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {"learning_rate": lr, "num_leaves": nl, "min_child_samples": mcs, "subsample": subs}
                        best_model = LGBMClassifier(
                            objective="binary" if len(np.unique(y)) == 2 else "multiclass",
                            n_estimators=1000,
                            random_state=SEED,
                            n_jobs=-1,
                            **best_params
                        )

    print(f"Best params: {best_params}")
    print(f"Best CV score: {best_score:.4f}")

    # 7) Final fit auf gesamten Daten
    best_model.fit(X_full, y)

    # 8) Artefakte speichern
    bundle = {
        "model": best_model,
        "num_imputer": num_imputer,
        "num_scaler": num_scaler,
        "embed_dim": args.embed_dim,
        "w2v_path": args.w2v_out,
        "user_id_order": train_user["user_id"].values
    }
    joblib.dump(bundle, args.model_out)
    print(f"Saved model to {args.model_out}")

    # OOF-Eval schnell
    oof_probs = np.zeros(len(X_full))
    oof_preds = np.empty(len(X_full), dtype=object)
    for trn_idx, val_idx in cv.split(X_full, y):
        model = LGBMClassifier(
            objective="binary" if len(np.unique(y)) == 2 else "multiclass",
            n_estimators=1000,
            random_state=SEED,
            n_jobs=-1,
            **best_params
        )
        model.fit(X_full[trn_idx], y[trn_idx])
        proba = model.predict_proba(X_full[val_idx])
        if len(np.unique(y)) == 2:
            pos = list(model.classes_).index(model.classes_[1])
            oof_probs[val_idx] = proba[:, pos]
        oof_preds[val_idx] = model.predict(X_full[val_idx])

    if len(np.unique(y)) == 2:
        y_bin = (y == model.classes_[1]).astype(int)
        print("OOF ROC-AUC:", roc_auc_score(y_bin, oof_probs))

    print("\nOOF classification report:")
    print(classification_report(y, oof_preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--model_out", type=str, default="models/w2v_lgbm_model.joblib")
    parser.add_argument("--w2v_out", type=str, default="artifacts/w2v.model")
    parser.add_argument("--embed_dim", type=int, default=64)
    args = parser.parse_args()
    main(args)
