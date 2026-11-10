from tkinter import *
from tkinter import messagebox
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from pathlib import Path

# Symptom names used by the original project
l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']

# Load the CSV files from the same folder as this Python file.
BASE_DIR = Path(__file__).resolve().parent
try:
    df = pd.read_csv(BASE_DIR / "Training.csv")
    test_df = pd.read_csv(BASE_DIR / "Testing.csv")
    missing_train = [c for c in l1 + ["prognosis"] if c not in df.columns]
    missing_test = [c for c in l1 + ["prognosis"] if c not in test_df.columns]
    if missing_train or missing_test:
        raise ValueError(
            f"Missing columns. Training.csv: {missing_train}; Testing.csv: {missing_test}"
        )
    X = df[l1].fillna(0)
    y = df["prognosis"].astype(str).str.strip()
    X_test = test_df[l1].fillna(0)
    y_test = test_df["prognosis"].astype(str).str.strip()
except Exception as exc:
    root = Tk()
    root.withdraw()
    messagebox.showerror("Project data error", str(exc))
    raise

# The original UI lets the user choose five symptoms.
root = Tk()
root.title("Disease Predictor using Machine Learning")
root.configure(background="blue")
root.geometry("1050x650")

Symptom1 = StringVar(value="Select symptom")
Symptom2 = StringVar(value="Select symptom")
Symptom3 = StringVar(value="Select symptom")
Symptom4 = StringVar(value="Select symptom")
Symptom5 = StringVar(value="Select symptom")
Name = StringVar()

Label(root, text="Disease Predictor using Machine Learning", fg="white", bg="blue",
      font=("Arial", 24, "bold")).grid(row=0, column=0, columnspan=3, padx=25, pady=20)

Label(root, text="Name of the Patient", fg="yellow", bg="black").grid(row=1, column=0, pady=10, sticky=W, padx=15)
Entry(root, textvariable=Name, width=35).grid(row=1, column=1, sticky=W)

symptom_vars = [Symptom1, Symptom2, Symptom3, Symptom4, Symptom5]
for i, var in enumerate(symptom_vars, start=2):
    Label(root, text=f"Symptom {i-1}", fg="yellow", bg="black").grid(row=i, column=0, pady=8, sticky=W, padx=15)
    OptionMenu(root, var, *sorted(l1)).grid(row=i, column=1, sticky=W)

result_vars = {}

def predict_disease(algorithm):
    try:
        selected = [var.get() for var in symptom_vars]
        if any(symptom == "Select symptom" or symptom not in l1 for symptom in selected):
            messagebox.showwarning("Missing symptoms", "Please select all five symptoms before predicting.")
            return
        if len(set(selected)) != len(selected):
            messagebox.showwarning("Duplicate symptoms", "Please select five different symptoms.")
            return

        # Build a fresh input row for every click; no old symptoms carry over.
        input_values = {symptom: 0 for symptom in l1}
        for symptom in selected:
            input_values[symptom] = 1
        input_row = pd.DataFrame([input_values], columns=l1)

        if algorithm == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
            output = result_vars[algorithm]
        elif algorithm == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            output = result_vars[algorithm]
        else:
            model = GaussianNB()
            output = result_vars[algorithm]

        model.fit(X, y)
        prediction = str(model.predict(input_row)[0])
        accuracy = accuracy_score(y_test, model.predict(X_test))

        output.config(state="normal")
        output.delete("1.0", END)
        output.insert(END, prediction)
        print(f"{algorithm} accuracy: {accuracy:.2%}")
        print(f"Patient: {Name.get().strip() or 'Not provided'} | Selected symptoms: {', '.join(selected)}")
    except Exception as exc:
        messagebox.showerror(f"{algorithm} prediction error", str(exc))
        print(f"{algorithm} prediction error: {exc}")


# Prediction buttons and output boxes
for row, algorithm in zip((8, 10, 12), ("Decision Tree", "Random Forest", "Naive Bayes")):
    Button(root, text=algorithm, command=lambda a=algorithm: predict_disease(a),
           bg="green", fg="yellow", width=18).grid(row=row, column=0, padx=15, pady=12, sticky=W)
    result = Text(root, height=2, width=45, bg="orange", fg="black", wrap=WORD)
    result.grid(row=row, column=1, padx=10, pady=12, sticky=W)
    result_vars[algorithm] = result

Label(root, text="Select five different symptoms, then click an algorithm button.",
      fg="white", bg="blue", font=("Arial", 10)).grid(row=14, column=0, columnspan=3, pady=15)

root.mainloop()
