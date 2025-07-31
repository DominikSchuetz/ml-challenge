# Gender Prediction – Exxeta ML Challenge

## 🧠 Zielsetzung

Auf Basis von **Surfverhalten (path, timestamp)** soll das **Geschlecht eines Users** vorhergesagt werden.  
Dazu stehen anonymisierte Klickdaten in `train.csv` und `test.csv` zur Verfügung.

Jede Zeile beschreibt:

- `user_id`: Nutzerkennung
- `path`: besuchte URL (verschlüsselt)
- `timestamp`: Zeitstempel des Besuchs
- `gender`: (nur in `train.csv`) tatsächliches Geschlecht

---

## 📁 Datenstruktur

```text
data/
├── train.csv     ← Trainingsdaten mit Labels
└── test.csv      ← Testdaten (nur user_id, path, timestamp)
```

---

## ⚙️ Modelle & Ansätze

| Modell                 | Features                                                             | ML-Modell        | Beschreibung |
|------------------------|----------------------------------------------------------------------|------------------|--------------|
| **1. Logistic Regression**  | TF-IDF (Term Frequency - Inverse Document Frequency) über Pfadfolge (1–2-gram), Zeit-Features (Stunde, Wochentag, Entropie, Klickanzahl etc.) | LogisticRegression(saga) | Sehr starke interpretable Baseline |
| **2. LightGBM + SVD**      | TF-IDF + TruncatedSVD (200–300 Komponenten) + Zeit-Features     | LightGBMClassifier | Kombination aus Vektorraum und Boosting |
| **3. Word2Vec + LightGBM** | Word2Vec-Embeddings über Pfad-Sequenz (User = avg(path_vecs)) + Zeit-Features | LightGBMClassifier | Modelliert Pfade als Token-Embeddings |

---

## 📈 Modelltraining & Validierung

Alle Modelle wurden per **Stratified K‑Fold Cross‑Validation** evaluiert.  
Metriken:

- **ROC-AUC** (Zielmetrik)
- F1-Score / Accuracy (unterstützend)

### ✅ Ergebnisse:

| Modell                   | ROC-AUC | F1-score (avg) | Anmerkung |
|--------------------------|---------|----------------|-----------|
| Logistic Regression      | 1.0000  | 1.00           | Basislösung mit TF-IDF + Zeitfeatures |
| LightGBM + SVD           | 1.0000  | 1.00           | Boosting auf reduzierter TF-IDF-Repräsentation |
| Word2Vec + LightGBM      | 0.99999 | 1.00           | Pfad-Embeddings mit Gensim-Word2Vec |

> Sehr hohe Scores, da die Datenstruktur stark geschlechtsspezifische Nutzungsmuster enthält.

---

## ▶️ Ausführung

### 1. Environment vorbereiten

```bash
python -m venv .venv
source .venv/bin/activate         # oder .venv\Scripts\Activate.ps1 unter Windows
pip install -r requirements.txt
```

### 2. Modell trainieren

#### Logistic Regression

```bash
python src/train.py --train_path data/train.csv --model_out models/logreg_model.joblib
```

#### LightGBM + SVD

```bash
python src/train_lgbm_svd.py --train_path data/train.csv --model_out models/lgbm_svd_model.joblib
```

#### Word2Vec + LightGBM

```bash
python src/train_w2v_lgbm.py --train_path data/train.csv --model_out models/w2v_lgbm_model.joblib --w2v_out artifacts/w2v.model --embed_dim 64
```

### 3. Vorhersagen erzeugen

#### Logistic Regression

```bash
python src/predict.py --test_path data/test.csv --model_path models/logreg_model.joblib --submission_out submission_logreg.csv
```

#### LightGBM + SVD

```bash
python src/predict_lgbm_svd.py --test_path data/test.csv --model_path models/lgbm_svd_model.joblib --submission_out submission_lgbm_svd.csv
```

#### Word2Vec + LightGBM

```bash
python src/predict_w2v_lgbm.py --test_path data/test.csv --model_path models/w2v_lgbm_model.joblib --submission_out submission_w2v_lgbm.csv
```

---

## 🏁 Output

Die `submission_*.csv` Dateien enthalten:

| user_id | prob_m | predicted_gender |
|---------|--------|------------------|
| ...     | 0.93   | m                |
| ...     | 0.14   | f                |

---


## 📄 Beispielhafte Ausgabe: Vorhersagen auf `test.csv`

Jedes Modell erzeugt eine Datei mit dem vorhergesagten Geschlecht pro `user_id`, z. B.:

### Logistic Regression (`submission.csv`)
| user_id                              |    prob_m | predicted_gender   |
|:-------------------------------------|----------:|:-------------------|
| 00122071-ddb4-4411-8c23-c7fee8072d0f | 0.127804  | f                  |
| 0013cfdf-b3fa-4c75-800a-b08f548ae9f1 | 0.0232429 | f                  |
| 00358796-31c0-4c4f-ab11-8a989a526ba8 | 0.887122  | m                  |

### LightGBM + SVD (`submission_lgbm_svd.csv`)
| user_id                              |      prob_m | predicted_gender   |
|:-------------------------------------|------------:|:-------------------|
| 00122071-ddb4-4411-8c23-c7fee8072d0f | 3.68217e-08 | f                  |
| 0013cfdf-b3fa-4c75-800a-b08f548ae9f1 | 9.0072e-09  | f                  |
| 00358796-31c0-4c4f-ab11-8a989a526ba8 | 1           | m                  |

### Word2Vec + LightGBM (`submission_w2v_lgbm.csv`)
| user_id                              |      prob_m | predicted_gender   |
|:-------------------------------------|------------:|:-------------------|
| 00122071-ddb4-4411-8c23-c7fee8072d0f | 1.05074e-07 | f                  |
| 0013cfdf-b3fa-4c75-800a-b08f548ae9f1 | 1.05074e-07 | f                  |
| 00358796-31c0-4c4f-ab11-8a989a526ba8 | 1           | m                  |

Alle Dateien enthalten die vorhergesagten Klassen (`predicted_gender`) und die zugehörige Wahrscheinlichkeit für die positive Klasse (z. B. `prob_m`).


## 💡 Fazit

- Der LogReg-Ansatz mit TF-IDF + Zeitverhalten ist überraschend stark – ein robuster Startpunkt für produktive Systeme.
- Die Kombination mit SVD + Boosting zeigt leichte Vorteile bei Flexibilität, ist aber speicherintensiver.
- Word2Vec ermöglicht semantischere Modellierung von Pfaden, performt ebenfalls sehr gut.

Für diese Challenge bietet die LogReg-basierte Lösung ein **exzellentes Preis-Leistungs-Verhältnis**.
