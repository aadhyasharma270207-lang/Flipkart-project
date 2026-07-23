import sys
# Hack to handle scikit-learn version mismatch where pickle looks for top-level '_loss' module
try:
    import sklearn._loss.loss as loss_mod
    sys.modules['_loss'] = loss_mod
except ImportError:
    pass

import pandas as pd
import numpy as np
import joblib
import os

model_path = r"c:\Users\dell\Desktop\flipkart-csat-deployment\flipkart_csat_model.pkl"
csv_path = r"c:\Users\dell\Desktop\flipkart-csat-deployment\Customer_support_data.csv"

# 1. Load model artifacts
artifacts = joblib.load(model_path)
model = artifacts['model']
scaler = artifacts['scaler']
selected_features = artifacts['selected_features']

print("Loaded Selected Features:", selected_features)

# 2. Replicate full preprocessing on the CSV
df = pd.read_csv(csv_path)

# Let's inspect target CSAT Score distribution
print("Unique CSAT Scores in CSV:", df['CSAT Score'].dropna().unique().tolist())

# Let's do the encoding on the original dataframe
cat_cols = ['channel_name', 'category', 'Tenure Bucket', 'Agent Shift']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

feature_cols = [c for c in df_encoded.columns if c in [
    'Item_price', 'connected_handling_time', 'response_time_minutes'
] or c.startswith(('channel_name_', 'category_', 'Tenure Bucket_', 'Agent Shift_'))]

X = df_encoded[feature_cols].copy()
X = X.fillna(X.median()).fillna(0)

# Check feature selection and transformation
X_transformed = X[selected_features].copy()
if 'response_time_minutes' in X_transformed.columns:
    X_transformed['response_time_minutes'] = np.log1p(X_transformed['response_time_minutes'])
else:
    if 'Item_price' in X_transformed.columns:
        X_transformed['Item_price'] = np.log1p(X_transformed['Item_price'])

# Let's select 3 specific rows of the original df that cover different classes/inputs
sample_indices = [10, 50, 100]  # Just 3 rows for verification
sample_rows_raw = df.iloc[sample_indices]

print("\n--- Verifying Preprocessing for 3 Sample Rows ---")

for idx in sample_indices:
    raw_row = df.iloc[idx]
    print(f"\nRow {idx} raw values:")
    print(f"  Item_price: {raw_row.get('Item_price')}")
    print(f"  channel_name: {raw_row.get('channel_name')}")
    print(f"  category: {raw_row.get('category')}")
    print(f"  Tenure Bucket: {raw_row.get('Tenure Bucket')}")
    print(f"  Agent Shift: {raw_row.get('Agent Shift')}")
    print(f"  Actual CSAT Score: {raw_row.get('CSAT Score')}")

    # Method A: get preprocessed values from the bulk dataframe
    preprocessed_A = X_transformed.loc[idx]
    scaled_A = scaler.transform([preprocessed_A.values])[0]
    pred_A = model.predict([scaled_A])[0]
    prob_A = model.predict_proba([scaled_A])[0]
    
    # Method B: manual mapping logic
    # We will build a single-row dict and then DataFrame with selected_features order.
    # Exclude drop_first values or undefined values.
    item_price = raw_row.get('Item_price')
    # Fill missing values if any
    if pd.isna(item_price):
        item_price = df['Item_price'].median()
        
    channel = raw_row.get('channel_name')
    category = raw_row.get('category')
    tenure = raw_row.get('Tenure Bucket')
    shift = raw_row.get('Agent Shift')

    # Construct single-row dictionary initialized to 0
    row_dict = {feat: 0.0 for feat in selected_features}
    
    # Item_price log-transformed
    row_dict['Item_price'] = np.log1p(float(item_price))
    
    # Map Channel
    if channel == 'Inbound':
        row_dict['channel_name_Inbound'] = 1.0
        
    # Map Category
    cat_col = f"category_{category}"
    if cat_col in row_dict:
        row_dict[cat_col] = 1.0
        
    # Map Tenure Bucket
    tenure_col = f"Tenure Bucket_{tenure}"
    if tenure_col in row_dict:
        row_dict[tenure_col] = 1.0
        
    # Map Agent Shift
    shift_col = f"Agent Shift_{shift}"
    if shift_col in row_dict:
        row_dict[shift_col] = 1.0

    # Build DataFrame
    df_single = pd.DataFrame([row_dict], columns=selected_features)
    scaled_B = scaler.transform(df_single)[0]
    pred_B = model.predict(scaled_B.reshape(1, -1))[0]
    prob_B = model.predict_proba(scaled_B.reshape(1, -1))[0]

    # Verify if features and scaling match
    features_match = np.allclose(preprocessed_A.values, df_single.values[0])
    scaled_match = np.allclose(scaled_A, scaled_B)
    pred_match = (pred_A == pred_B)
    
    print(f"  Features Match: {features_match}")
    if not features_match:
        print("  Method A features:", preprocessed_A.to_dict())
        print("  Method B features:", df_single.iloc[0].to_dict())
    print(f"  Scaled Values Match: {scaled_match}")
    print(f"  Prediction Match: {pred_match} (A: {pred_A}, B: {pred_B})")
    print(f"  Probability Match: {np.allclose(prob_A, prob_B)} (A: {prob_A}, B: {prob_B})")
