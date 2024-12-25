import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model():
    # Load data
    data = pd.read_csv("data/processed/processed_data.csv")
    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Simpan model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("Model telah dilatih dan disimpan.")

if __name__ == "__main__":
    train_model()
