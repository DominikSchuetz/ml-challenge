import pandas as pd
import numpy as np
from scipy.stats import entropy

def _time_deltas(ts: pd.Series):
    ts = ts.sort_values().values.astype("datetime64[ns]").astype("int64") / 1e9  # seconds
    if len(ts) <= 1:
        return np.nan, np.nan, np.nan
    diffs = np.diff(ts)
    return float(np.mean(diffs)), float(np.median(diffs)), float(np.std(diffs))

def _path_entropy(paths: pd.Series):
    counts = paths.value_counts(normalize=True).values
    return float(entropy(counts, base=2)) if len(counts) > 0 else 0.0

def aggregate_user_level(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Erwartet ein DataFrame mit Spalten: user_id, path, timestamp [, gender]
    Gibt ein DataFrame auf user_id-Ebene mit:
      - doc (space-separierte Pfade als 'Dokument')
      - numerischen Features (zeitliche Histogramme, diversitätsmaße, etc.)
      - gender (nur bei train)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["user_id", "timestamp"])

    # Stunden- & Wochentags-Histogramme
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek

    # Aggregationen
    agg_rows = []
    for uid, g in df.groupby("user_id", sort=False):
        paths_sorted = g.sort_values("timestamp")["path"].astype(str)
        doc = " ".join(paths_sorted.tolist())

        n_visits = len(g)
        n_unique_paths = g["path"].nunique()
        mean_dt, median_dt, std_dt = _time_deltas(g["timestamp"])
        path_ent = _path_entropy(g["path"])

        # Normalisierte Histogramme
        hour_hist = np.bincount(g["hour"].fillna(0).astype(int), minlength=24)
        hour_hist = hour_hist / max(1, hour_hist.sum())

        dow_hist = np.bincount(g["dow"].fillna(0).astype(int), minlength=7)
        dow_hist = dow_hist / max(1, dow_hist.sum())

        row = {
            "user_id": uid,
            "doc": doc,
            "n_visits": n_visits,
            "n_unique_paths": n_unique_paths,
            "ratio_unique": n_unique_paths / n_visits if n_visits > 0 else 0.0,
            "mean_dt": mean_dt,
            "median_dt": median_dt,
            "std_dt": std_dt,
            "path_entropy": path_ent,
        }

        # add hour and dow hist
        for h in range(24):
            row[f"hour_{h:02d}"] = hour_hist[h]
        for d in range(7):
            row[f"dow_{d}"] = dow_hist[d]

        if is_train and "gender" in g.columns:
            # Annahme: ein User hat genau ein Gender-Label
            row["gender"] = g["gender"].iloc[0]

        agg_rows.append(row)

    user_df = pd.DataFrame(agg_rows)
    return user_df


def read_and_prepare_train(train_path: str) -> pd.DataFrame:
    df = pd.read_csv(train_path)
    return aggregate_user_level(df, is_train=True)


def read_and_prepare_test(test_path: str) -> pd.DataFrame:
    df = pd.read_csv(test_path)
    return aggregate_user_level(df, is_train=False)


# Für Word2Vec (Sequenzen je User)
def get_user_path_sequences(csv_path: str, has_gender: bool) -> pd.DataFrame:
    """
    Gibt ein DataFrame mit Spalten:
      - user_id
      - paths: Liste[str] (zeitlich sortierte Pfade)
      - gender (optional)
    """
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["user_id", "timestamp"])

    rows = []
    for uid, g in df.groupby("user_id", sort=False):
        paths = g["path"].astype(str).tolist()
        row = {"user_id": uid, "paths": paths}
        if has_gender and "gender" in g.columns:
            row["gender"] = g["gender"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)