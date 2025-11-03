import pandas as pd, argparse, joblib
def predict(model_path, input_csv, out_csv=None):
    art = joblib.load(model_path)
    model = art['model']
    feats = art['features']
    df = pd.read_csv(input_csv)
    X = df[feats]
    preds = model.predict(X)
    df['predicted_lead_time'] = preds
    if out_csv:
        df.to_csv(out_csv, index=False)
        print('Saved predictions to', out_csv)
    else:
        print(df[['order_id','predicted_lead_time']].head())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    predict(args.model, args.input, args.out)
