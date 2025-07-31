import argparse
import joblib
import pandas as pd

from prepare_data import read_and_prepare_test

def main(args):
    # 1) Test aggregieren
    test_user = read_and_prepare_test(args.test_path)

    # 2) Modell laden
    model = joblib.load(args.model_path)

    # 3) Vorhersagen
    proba = model.predict_proba(test_user)
    classes = model.classes_  # z.B. ['female','male'] – Reihenfolge wichtig

    # 4) Submission bauen
    #    Wähle, welche Spalte du abgeben willst – oft sinnvoll: P(male)
    #    Wir nehmen hier der Einfachheit halber die zweite Klasse (classes_[1]).
    positive_class = classes[1]
    positive_idx = list(classes).index(positive_class)

    submission = pd.DataFrame({
        "user_id": test_user["user_id"],
        f"prob_{positive_class}": proba[:, positive_idx],
        "predicted_gender": classes[proba.argmax(axis=1)]
    })

    submission.to_csv(args.submission_out, index=False)
    print(f"Saved predictions to {args.submission_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str, default="data/test.csv")
    parser.add_argument("--model_path", type=str, default="models/model.joblib")
    parser.add_argument("--submission_out", type=str, default="submission.csv")
    args = parser.parse_args()
    main(args)
