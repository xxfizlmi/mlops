import joblib
import os

def deploy_model():
    model = joblib.load("models/model.pkl")

    # Simpan model
    os.makedirs("models/deployed", exist_ok=True)
    joblib.dump(model, "models/deployed/deployed_model.pkl")
    print("Model telah disiapkan untuk deployment.")

if __name__ == "__main__":
    deploy_model()
