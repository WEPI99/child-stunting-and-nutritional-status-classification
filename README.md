# Posyandu AI

A small Flask API that screens toddlers (0–60 months) for **stunting** and **weight-based nutritional status** using two SVM classifiers (scikit-learn).

> Screening aid only, not a medical diagnosis. Results should be confirmed by a health professional.

## How It Works

```
JSON request ──► POST /predict ──┬─► stunted.pkl  (age, height, sex) ─► stunted / normal
                                 └─► nutrion.pkl  (age, weight, sex) ─► underweight / normal / overweight
                                                     └─► combined into one result (JSON)
```

| Model | Features (in this order) | Classes |
|-------|--------------------------|---------|
| `stunted.pkl` | Age (months), Height (cm), Sex | 0 severely stunted, 1 stunted, 2 normal, 3 tall |
| `nutrion.pkl` | Age (months), Weight (kg), Sex | 0 severely underweight, 1 underweight, 2 normal, 3 overweight |

Both are pipelines: `StandardScaler` → `SVC(kernel="rbf", C=10)`. Sex is encoded as `male = 0`, `female = 1`.

**Result mapping:** stunting classes 0–1 → `stunted`, 2–3 → `normal`; nutrition classes 0–1 → `underweight`, 2 → `normal`, 3 → `overweight`. The nutrition result is applied last, so it takes priority over the stunting result.

**Training pipeline** (see the two notebooks): load CSV → encode sex/labels → drop missing and duplicate rows → 80/20 split → 5-fold cross-validation → evaluate on the test set → save with `joblib`.

| Model | Rows after de-duplication | CV accuracy | Test accuracy |
|-------|---------------------------|-------------|---------------|
| Stunting | 34,921 (from 120,999) | 98.58% | 98.83% |
| Nutrition | 2,372 (from 122,122) | 83.34% | 82.11% |

Duplicates are removed so identical rows cannot appear in both the train and test sets. The datasets are labeled with threshold rules, so the stunting model mostly learns that rule; its high accuracy reflects that, not clinical validation. Nutrition labels are not fully determined by age, weight and sex alone, hence the lower accuracy.

## Installation

Requires Python 3.10+.

```bash 
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Keep the same scikit-learn version for training and serving, since `.pkl` files are version-sensitive.

## Run

```bash
flask --app app run --port 5001
```

> On macOS, port 5000 is taken by AirPlay Receiver (it returns HTTP 403), so use another port such as 5001.

## API

`POST /predict` with `Content-Type: application/json`

| Field    | Type | Description    |
|----------|------|----------------|
| `months` | number | Age in months  |
| `gender` | string | Male or Female |
| `bb`     | number | Weight (kg)    |
| `tb`     | number | Height (cm)    |

Request:

```bash
curl -X POST http://127.0.0.1:5001/predict \
  -H "Content-Type: application/json" \
  -d '{"months": 45, "gender": "Male", "bb": 12.5, "tb": 95}'
```

Response:

```json
{
  "data": "normal",
  "message": "data has been generated successfully"
}
```

`data` is one of `stunted`, `underweight`, `normal`, `overweight`.