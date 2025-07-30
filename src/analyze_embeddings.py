import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

from prepare_data import get_user_path_sequences


def avg_embed(paths, w2v, dim):
    vecs = [w2v.wv[p] for p in paths if p in w2v.wv.key_to_index]
    if len(vecs) == 0:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Lade Daten & Word2Vec
    print("Loading data & Word2Vec …")
    seq_df = get_user_path_sequences(args.train_path, has_gender=True)
    w2v = Word2Vec.load(args.w2v_path)
    dim = w2v.vector_size

    # 2) Baue User-Embeddings
    print("Building user embeddings …")
    user_embeds = np.vstack([avg_embed(paths, w2v, dim) for paths in seq_df["paths"]])
    seq_df["embed_idx"] = range(len(seq_df))  # zur Sicherheit

    # 3) Klassen-Zentroiden
    print("Computing class centroids …")
    male_mask = seq_df["gender"].astype(str).str.lower().str.startswith("m")
    female_mask = seq_df["gender"].astype(str).str.lower().str.startswith("f")

    male_centroid = user_embeds[male_mask].mean(axis=0)
    female_centroid = user_embeds[female_mask].mean(axis=0)

    # 4) Score pro Path-Token: Cosine-Differenz zu Zentroiden
    print("Scoring paths …")
    all_tokens = list(w2v.wv.index_to_key)
    token_vecs = np.vstack([w2v.wv[t] for t in all_tokens])

    male_cos = cosine_similarity(token_vecs, male_centroid.reshape(1, -1)).ravel()
    female_cos = cosine_similarity(token_vecs, female_centroid.reshape(1, -1)).ravel()

    diff = male_cos - female_cos  # >0: eher männlich, <0: eher weiblich
    abs_diff = np.abs(diff)

    token_df = pd.DataFrame({
        "token": all_tokens,
        "cos_male": male_cos,
        "cos_female": female_cos,
        "cos_diff": diff,
        "abs_diff": abs_diff
    })

    # 5) Top-K Pfade je Klasse
    top_k = args.top_k
    top_male = token_df.sort_values("cos_diff", ascending=False).head(top_k)
    top_female = token_df.sort_values("cos_diff", ascending=True).head(top_k)

    top_male.to_csv(outdir / f"top_{top_k}_male_paths.csv", index=False)
    top_female.to_csv(outdir / f"top_{top_k}_female_paths.csv", index=False)

    print("\n=== Top männlich kodierte Pfade (cos_male - cos_female groß) ===")
    print(top_male.head(20).to_string(index=False))

    print("\n=== Top weiblich kodierte Pfade (cos_female - cos_male groß) ===")
    print(top_female.head(20).to_string(index=False))

    # 6) Visualisierung: PCA auf Path-Embeddings
    print("\nRunning PCA for visualization (paths) …")
    pca = PCA(n_components=2, random_state=42)
    token_2d = pca.fit_transform(token_vecs)

    # Farbwert = diff (positiv -> männlich, negativ -> weiblich)
    plt.figure()
    plt.scatter(token_2d[:, 0], token_2d[:, 1], c=diff, s=5)
    plt.title("Path-Embeddings (PCA), color = cos(male) - cos(female)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(outdir / "paths_pca.png", dpi=200)

    # Optional: die Top-K labeln
    def annotate_points(df_sub, color='black'):
        for _, row in df_sub.iterrows():
            idx = all_tokens.index(row["token"])
            x, y = token_2d[idx]
            plt.annotate(row["token"], (x, y), fontsize=6)

    plt.figure()
    plt.scatter(token_2d[:, 0], token_2d[:, 1], c=diff, s=5)
    annotate_points(top_male, color='black')
    annotate_points(top_female, color='black')
    plt.title(f"Top {top_k} gender-discriminative paths annotated")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(outdir / f"paths_pca_top{top_k}_annotated.png", dpi=200)

    # 7) Visualisierung: PCA auf User-Embeddings
    print("Running PCA for visualization (users) …")
    user_pca = PCA(n_components=2, random_state=42)
    user_2d = user_pca.fit_transform(user_embeds)

    plt.figure()
    # 0/1 codieren
    y_bin = np.where(male_mask, 1, 0)
    plt.scatter(user_2d[y_bin == 0, 0], user_2d[y_bin == 0, 1], s=5, label="female")
    plt.scatter(user_2d[y_bin == 1, 0], user_2d[y_bin == 1, 1], s=5, label="male")
    plt.legend()
    plt.title("User-Embeddings (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(outdir / "users_pca.png", dpi=200)

    print(f"\nSaved:\n- {outdir / f'top_{top_k}_male_paths.csv'}"
          f"\n- {outdir / f'top_{top_k}_female_paths.csv'}"
          f"\n- {outdir / 'paths_pca.png'}"
          f"\n- {outdir / f'paths_pca_top{top_k}_annotated.png'}"
          f"\n- {outdir / 'users_pca.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--w2v_path", type=str, default="artifacts/w2v.model")
    parser.add_argument("--outdir", type=str, default="artifacts/embedding_analysis")
    parser.add_argument("--top_k", type=int, default=30)
    args = parser.parse_args()
    main(args)
