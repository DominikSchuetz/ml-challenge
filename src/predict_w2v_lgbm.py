import argparse
import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from prepare_data import read_and_prepare_test, get_user_path_sequences

def average_embedding(paths, w2v_model, embed_dim):
    vecs = []
    for p in paths:
        if p in w2v_model.wv:
            vecs.append(w2v_model.wv[p])
    if len(vecs) == 0:
        return np.zeros(embed_dim, dtype=np.float32)
    return np.mean(vecs, axis=0)

def main(args):
    # 1) Lade Bundle
    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    num_imputer = bundle["num_imputer"]
    num_scaler = bundle["num_scaler"]
    embed_dim = bundle["embed_dim"]
    w2v = Word2Vec.load(bundle["w2v_path"])

    # 2) Aggregierte numerische Features
    test_user = read_and_prepare_test(args.test_path)
    X_num = test_user.drop(columns=["user_id", "doc"])

    # 3) Sequenzen
    seq_df = get_user_path_sequences(args.test_path, has_gender=False)

    # 4) Embeddings
    user_embeds = []
    for paths in seq_df["paths"]:
        user_embeds.append(average_embedding(paths, w2v, embed_dim))
    user_embeds = np.vstack(user_embeds)

    # 5) Numerik vorbereiten
    X_num_imp = num_imputer.transform(X_num)
    X_num_scaled = num_scaler.transform(X_num_imp)

    X_full = np.hstack([user_embeds, X_num_scaled])

    proba = model.predict_proba(X_full)
    classes = model.classes_
    pos_idx = 1 if len(classes) == 2 else None

    out = pd.DataFrame({"user_id": test_user["user_id"]})
    if pos_idx is not None:
        out[f"prob_{classes[pos_idx]}"] = proba[:, pos_idx]
    else:
        out["max_prob"] = proba.max(axis=1)
    out["predicted_gender"] = classes[proba.argmax(axis=1)]

    out.to_csv(args.submission_out, index=False)
    print(f"Saved predictions to {args.submission_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default="data/test.csv")
    parser.add_argument("--model_path", type=str, default="models/w2v_lgbm_model.joblib")
    parser.add_argument("--submission_out", type=str, default="submission_w2v_lgbm.csv")
    args = parser.parse_args()
    main(args)
