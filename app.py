"""
app.py
------
Flask website for the Diabetes Risk Prediction System.

Run train_model.py first (it creates the model and metrics files),
then start the website with:   python app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = BASE_DIR / "model" / "models.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

app = Flask(__name__)

# The order of this list must match the training order in train_model.py.
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Form field definitions: label, unit, allowed range, and whether whole numbers are required.
# "min_exclusive" means the value must be greater than the minimum (zero is not a real
# measurement for these fields).
FIELDS = [
    {"name": "Pregnancies", "label": "Pregnancies", "unit": "count", "min": 0, "max": 20,
     "integer": True, "min_exclusive": False, "step": "1",
     "help": "Number of times pregnant (enter 0 if none)."},
    {"name": "Glucose", "label": "Glucose", "unit": "mg/dL", "min": 0, "max": 600,
     "integer": False, "min_exclusive": True, "step": "any",
     "help": "Plasma glucose concentration from a lab test."},
    {"name": "BloodPressure", "label": "Blood pressure", "unit": "mmHg", "min": 0, "max": 200,
     "integer": False, "min_exclusive": True, "step": "any",
     "help": "Diastolic blood pressure (the lower number)."},
    {"name": "SkinThickness", "label": "Skin thickness", "unit": "mm", "min": 0, "max": 100,
     "integer": False, "min_exclusive": True, "step": "any",
     "help": "Triceps skin fold thickness."},
    {"name": "Insulin", "label": "Insulin", "unit": "µU/mL", "min": 0, "max": 900,
     "integer": False, "min_exclusive": True, "step": "any",
     "help": "2-hour serum insulin."},
    {"name": "BMI", "label": "BMI", "unit": "kg/m²", "min": 0, "max": 80,
     "integer": False, "min_exclusive": True, "step": "any",
     "help": "Body Mass Index = weight (kg) / height (m) squared."},
    {"name": "DiabetesPedigreeFunction", "label": "Diabetes pedigree function", "unit": "score",
     "min": 0, "max": 3, "integer": False, "min_exclusive": True, "step": "any",
     "help": "Dataset score estimating diabetes likelihood from family history (typical range 0.08 to 2.5)."},
    {"name": "Age", "label": "Age", "unit": "years", "min": 0, "max": 120,
     "integer": True, "min_exclusive": True, "step": "1",
     "help": "Age in whole years. The model was trained on people aged 21 and above."},
]

# Load the trained model once when the app starts.
models = {}
model_error = None
try:
    models = joblib.load(MODELS_PATH)
except FileNotFoundError:
    model_error = "Trained models were not found. Please run  python train_model.py  first."
except Exception as error:  # pragma: no cover - unexpected problems
    model_error = f"The trained model could not be loaded: {error}"


def validate_form(form):
    """Check every field. Returns (values_dict, errors_dict)."""
    values = {}
    errors = {}

    for field in FIELDS:
        name = field["name"]
        raw = (form.get(name) or "").strip()

        if raw == "":
            errors[name] = f"{field['label']} is required."
            continue

        try:
            number = float(raw)
        except ValueError:
            errors[name] = f"{field['label']} must be a number."
            continue

        if number != number or number in (float("inf"), float("-inf")):
            errors[name] = f"{field['label']} must be a valid number."
            continue

        if number < 0:
            errors[name] = f"{field['label']} cannot be negative."
            continue

        if field["min_exclusive"] and number <= field["min"]:
            errors[name] = f"{field['label']} must be greater than {field['min']}."
            continue

        if number > field["max"]:
            errors[name] = f"{field['label']} must not be more than {field['max']} {field['unit']}."
            continue

        if field["integer"] and number != int(number):
            errors[name] = f"{field['label']} must be a whole number."
            continue

        values[name] = number

    return values, errors


def get_recommendations(is_higher_risk):
    if is_higher_risk:
        return [
            "Consult a doctor or other healthcare professional soon.",
            "Discuss appropriate diabetes tests (for example fasting glucose or HbA1c) with them.",
            "Bring your measured values and any family history of diabetes to the appointment.",
            "Until then, keep a balanced diet, limit sugary drinks and stay physically active.",
        ]
    return [
        "Continue routine health check-ups and screening as advised by your doctor.",
        "Eat a balanced diet with plenty of vegetables, whole grains and fibre.",
        "Aim for regular physical activity (for example, about 150 minutes of brisk walking a week).",
        "Maintain a healthy weight and avoid smoking.",
        "See a doctor if you notice symptoms such as constant thirst, frequent urination or unusual tiredness.",
    ]


@app.route("/")
def home():
    return render_template("index.html", fields=FIELDS, values={}, errors={}, model_error=model_error)


@app.route("/predict", methods=["POST"])
def predict():
    if not models:
        return render_template("message.html", title="Model not available", message=model_error), 503

    values, errors = validate_form(request.form)
    if errors:
        # Show the form again with the user's input and the error messages.
        return render_template(
            "index.html", fields=FIELDS, values=request.form, errors=errors, model_error=None
        ), 400

    # Build one row with the columns in the same order used for training.
    row = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)

    try:
        model_predictions = []
        for algorithm_name, trained_model in models.items():
            probability = float(trained_model.predict_proba(row)[0][1])
            prediction = int(trained_model.predict(row)[0])
            model_predictions.append({
                "algorithm": algorithm_name,
                "prediction": "Higher risk" if prediction == 1 else "Lower risk",
                "score": probability,
                "is_positive": prediction == 1,
            })
        # The app uses the highest-accuracy model selected during training.
        metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        best_algorithm = metrics_data["best_algorithm"]
        best_result = next(item for item in model_predictions if item["algorithm"] == best_algorithm)
        score = best_result["score"]
        is_higher_risk = best_result["is_positive"]
    except Exception as error:
        return render_template(
            "message.html", title="Prediction failed",
            message=f"Something went wrong while making the prediction: {error}"
        ), 500

    age_warning = values["Age"] < 21
    shown_values = [
        {"label": f["label"], "unit": f["unit"],
         "value": int(values[f["name"]]) if f["integer"] else values[f["name"]]}
        for f in FIELDS
    ]

    return render_template(
        "result.html",
        is_higher_risk=is_higher_risk,
        score=score,
        best_algorithm=best_algorithm,
        model_predictions=model_predictions,
        recommendations=get_recommendations(is_higher_risk),
        shown_values=shown_values,
        age_warning=age_warning,
    )


@app.route("/accuracy")
def accuracy():
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return render_template(
            "message.html", title="Metrics not available",
            message="The metrics file was not found. Please run  python train_model.py  first."
        ), 404
    except Exception as error:
        return render_template(
            "message.html", title="Metrics could not be read", message=str(error)
        ), 500
    return render_template("accuracy.html", m=metrics)


@app.route("/about")
def about():
    return render_template("about.html")


@app.errorhandler(404)
def not_found(_error):
    return render_template(
        "message.html", title="Page not found", message="The page you asked for does not exist."
    ), 404


if __name__ == "__main__":
    if model_error:
        print("WARNING:", model_error)
    app.run(host="127.0.0.1", port=5000, debug=False)
