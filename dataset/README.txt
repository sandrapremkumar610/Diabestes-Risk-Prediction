DATASET FOLDER
==============

This folder holds diabetes.csv, the Pima Indians Diabetes Dataset.

You do not need to download it by hand. When you run

    python train_model.py

the script downloads the file into this folder if it is missing.

If your internet connection or firewall blocks the download, get the CSV
yourself (search for "Pima Indians Diabetes Dataset CSV", for example on Kaggle
or the UCI/GitHub mirrors) and save it here as:

    dataset/diabetes.csv

The file must have this header row:

    Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome

About the data
--------------
- 768 records, 8 input features, 1 target (Outcome: 0 = non-diabetic, 1 = diabetic).
- Records come from Pima Indian women aged 21 and above, so the data does not
  represent every population.
- In Glucose, BloodPressure, SkinThickness, Insulin and BMI, a value of 0 means
  "not measured". train_model.py treats these zeros as missing values.
