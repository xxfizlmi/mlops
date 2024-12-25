# Makefile untuk pipeline machine learning dengan virtual environment otomatis (Windows)

# Variabel untuk virtual environment
VENV = .venv
PYTHON = $(VENV)\Scripts\python
PIP = $(VENV)\Scripts\pip

# Pastikan virtual environment terinstal sebelum langkah apa pun
.PHONY: venv
venv:
	@if not exist $(VENV) ($(PYTHON) -m venv $(VENV))
	@$(PIP) install --upgrade pip
	@$(PIP) install pandas numpy scikit-learn joblib

# Unduh dan proses data
.PHONY: data
data: venv
	@$(PYTHON) scripts/data_prep.py

# Latih model
.PHONY: train
train: venv
	@$(PYTHON) scripts/train_model.py

# Evaluasi model
.PHONY: evaluate
evaluate: venv
	@$(PYTHON) scripts/evaluate_model.py

# Simpan model untuk deployment
.PHONY: deploy
deploy: venv
	@$(PYTHON) scripts/deploy_model.py

# Hapus virtual environment (opsional untuk membersihkan lingkungan)
.PHONY: clean
clean:
	@rmdir /s /q $(VENV)
	@echo "Virtual environment telah dihapus."
