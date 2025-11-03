# Supply Chain Lead-Time Prediction (Full Demo - Small)

This project contains a synthetic dataset (~1000 shipments) and a ready-to-run pipeline
to preprocess data, train a RandomForest regression model to predict shipment lead time,
evaluate it, and run sample predictions.

**To run locally (recommended):**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/preprocess.py --raw data/shipments_synthetic_1000.csv --out data/processed/features.csv
python src/train_model.py --features data/processed/features.csv --out experiments/rf_model.joblib
python src/evaluate.py --model experiments/rf_model.joblib --features data/processed/features.csv
python src/predict_sample.py --model experiments/rf_model.joblib --input data/processed/features.csv --out predictions.csv
```

Files included:
- `data/shipments_synthetic_1000.csv` : synthetic raw dataset
- `src/preprocess.py` : preprocessing and feature engineering
- `src/train_model.py` : train RandomForest model and save artifact
- `src/evaluate.py` : evaluate saved model (MAE, RMSE)
- `src/predict_sample.py` : run predictions on CSV input
- `notebooks/01_demo.ipynb` : demo notebook (loads data and trains quickly)
- `experiments/rf_model.joblib` : trained model (already included)
- `requirements.txt` : python dependencies
