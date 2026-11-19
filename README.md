# Diabetes Risk Prediction System Using Four Machine Learning Algorithms

An MCA educational project using Flask, Pandas and Scikit-learn. The system trains and compares Logistic Regression, Decision Tree, Random Forest and Support Vector Machine (SVM) on the Pima Indians Diabetes dataset. It shows their test metrics in a comparison dashboard and displays each model's prediction for a submitted form.

> Educational demonstration only. This project is not clinically validated and cannot diagnose or rule out diabetes.

## Technologies
Python, Flask, Pandas, NumPy, Scikit-learn, Joblib, HTML, CSS, Chart.js.

## Run in VS Code

1. Extract the ZIP and open the `Diabetes_Risk_Prediction` folder.
2. Open VS Code terminal in that folder.
3. Install packages:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Train and compare the four models:

   ```bash
   python train_model.py
   ```

5. Start the website:

   ```bash
   python app.py
   ```

6. Open `http://127.0.0.1:5000`.
7. Select **Model Accuracy** to view the comparison chart and metrics table.

## Four algorithms

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

All algorithms use the same stratified 80% training / 20% testing split. Preprocessing is included inside each Scikit-learn Pipeline so median imputation and scaling are fitted using training data only. The models are compared using accuracy, precision, recall, F1 score and ROC-AUC. The website selects the model with the highest held-out test accuracy, while also displaying the other metrics because accuracy alone is not sufficient for health-related classification.

## Project structure

```text
Diabetes_Risk_Prediction/
├── app.py
├── train_model.py
├── dataset/
│   ├── diabetes.csv
│   └── README.txt
├── model/                  # generated when train_model.py runs
│   ├── models.pkl
│   └── metrics.json
├── notebooks/
│   └── data_analysis.ipynb
├── static/
│   └── style.css
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   ├── about.html
│   ├── accuracy.html
│   └── message.html
├── requirements.txt
└── README.md
```

`train_model.py` creates `model/models.pkl` and `model/metrics.json`. Run it again whenever you change the dataset or training settings. The Accuracy page reads the newly calculated values from `metrics.json`; no sample accuracy values are hardcoded.

The accuracy graph uses Chart.js from a CDN, so an internet connection is needed when loading that page unless Chart.js is downloaded and served locally.
