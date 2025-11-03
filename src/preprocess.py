import pandas as pd, argparse, os

def preprocess(df):
    df['order_date'] = pd.to_datetime(df['order_date'])
    if 'actual_delivery_date' in df.columns:
        df['actual_delivery_date'] = pd.to_datetime(df['actual_delivery_date'])
        df['lead_time_days'] = (df['actual_delivery_date'] - df['order_date']).dt.days
    if 'requested_delivery_date' in df.columns:
        df['requested_delivery_date'] = pd.to_datetime(df['requested_delivery_date'])
        df['requested_lead_days'] = (df['requested_delivery_date'] - df['order_date']).dt.days
    df['order_weekday'] = df['order_date'].dt.weekday
    df['order_month'] = df['order_date'].dt.month
    for c in ['weight','volume']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(df[c].median())
    cat_cols = ['origin','destination','carrier']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna('UNK').astype(str)
            df[c+'_enc'] = df[c].astype('category').cat.codes
    feats = ['weight','volume','requested_lead_days','order_weekday','order_month']
    for c in cat_cols:
        feats.append(c+'_enc')
    feats = [f for f in feats if f in df.columns]
    out = df[['order_id'] + feats + ['lead_time_days']].copy()
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.raw)
    out = preprocess(df)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print('Saved processed features to', args.out)
