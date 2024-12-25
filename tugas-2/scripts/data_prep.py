import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def download_and_process_data():
    # Unduh dataset
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    data = pd.read_csv(url)

    # Normalisasi fitur
    scaler = StandardScaler()
    data.iloc[:, :-1] = scaler.fit_transform(data.iloc[:, :-1])

    # Simpan data
    os.makedirs("data/processed", exist_ok=True)
    data.to_csv("data/processed/processed_data.csv", index=False)
    print("Data telah diunduh dan diproses.")

if __name__ == "__main__":
    download_and_process_data()
