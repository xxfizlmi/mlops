import os
import time
import pickle
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import tensorflow as tf

# Buat folder untuk menyimpan output
os.makedirs("models", exist_ok=True)

# Load dataset
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Serialize with Pickle
start_time = time.time()
with open("models/model_pickle.pkl", "wb") as f:
    pickle.dump(model, f)
pickle_save_time = time.time() - start_time

start_time = time.time()
with open("models/model_pickle.pkl", "rb") as f:
    loaded_model_pickle = pickle.load(f)
pickle_load_time = time.time() - start_time

# Serialize with Joblib
start_time = time.time()
joblib.dump(model, "models/model_joblib.pkl")
joblib_save_time = time.time() - start_time

start_time = time.time()
loaded_model_joblib = joblib.load("models/model_joblib.pkl")
joblib_load_time = time.time() - start_time

# Serialize with ONNX
initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

start_time = time.time()
with open("models/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
onnx_save_time = time.time() - start_time

start_time = time.time()
onnx_model_loaded = onnx_model.SerializeToString()
onnx_load_time = time.time() - start_time

# Serialize with TensorFlow
start
