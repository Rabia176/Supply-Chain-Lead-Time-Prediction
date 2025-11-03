import pandas as pd, argparse, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def train(features_csv, out_model):
    df = pd.read_csv(features_csv)
    df = df.dropna(subset=['lead_time_days'])
    X = df.drop(columns=['order_id','lead_time_days'])
    y = df['lead_time_days']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('MAE:', mean_absolute_error(y_test, preds))
    print('RMSE:', mean_squared_error(y_test, preds, squared=False))
    os.makedirs(os.path.dirname(out_model), exist_ok=True)
    joblib.dump({'model': model, 'features': X.columns.tolist()}, out_model)
    print('Saved model to', out_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    train(args.features, args.out)
