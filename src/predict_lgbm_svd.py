import argparse
import joblib
import pandas as pd
from prepare_data import read_and_prepare_test

def main(args):
    test_user = read_and_prepare_test(args.test_path)
    model = joblib.load(args.model_path)

    proba = model.predict_proba(test_user)
    classes = model.named_steps["clf"].classes_
    pos_idx = 1 if len(classes) == 2 else None

    df = pd.DataFrame({"user_id": test_user["user_id"]})
    if pos_idx is not None:
        df[f"prob_{classes[pos_idx]}"] = proba[:, pos_idx]
    else:
        # Multiclass: gib max-Proba und predicted label aus
        df["max_prob"] = proba.max(axis=1)
    df["predicted_gender"] = classes[proba.argmax(axis=1)]

    df.to_csv(args.submission_out, index=False)
    print(f"Saved predictions to {args.submission_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default="data/test.csv")
    parser.add_argument("--model_path", type=str, default="models/lgbm_svd_model.joblib")
    parser.add_argument("--submission_out", type=str, default="submission_lgbm_svd.csv")
    args = parser.parse_args()
    main(args)
