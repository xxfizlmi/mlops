import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib

def evaluate_model():
    # Load data dan model
    data = pd.read_csv("data/processed/processed_data.csv")
    model = joblib.load("models/model.pkl")

    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    y_pred = model.predict(X)

    # Evaluasi performa
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)

    # Simpan hasil evaluasi
    with open("results/evaluation.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n\n{report}")
    print("Hasil evaluasi disimpan di results/evaluation.txt.")

if __name__ == "__main__":
    evaluate_model()
