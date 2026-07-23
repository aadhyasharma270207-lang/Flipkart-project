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

# Load artifacts
artifacts = joblib.load("flipkart_csat_model.pkl")
model = artifacts['model']
scaler = artifacts['scaler']
selected_features = artifacts['selected_features']

# Define combinations
combinations = [
    {
        "name": "Satisfied Combination (Row 0)",
        "Item_price": 979.0,
        "Channel": "Outcall",
        "Category": "Product Queries",
        "Tenure": "On Job Training",
        "Shift": "Morning"
    },
    {
        "name": "Unsatisfied Combination (Row 238)",
        "Item_price": 19999.0,
        "Channel": "Inbound",
        "Category": "Cancellation",
        "Tenure": "0-30",
        "Shift": "Morning"
    }
]

for comb in combinations:
    print(f"\n--- Testing: {comb['name']} ---")
    
    # 1. Construct dict
    row_dict = {feat: 0.0 for feat in selected_features}
    
    # Item_price log-transformed
    row_dict['Item_price'] = np.log1p(float(comb['Item_price']))
    
    # Channel
    if comb['Channel'] == 'Inbound':
        row_dict['channel_name_Inbound'] = 1.0
        
    # Category
    cat_col = f"category_{comb['Category']}"
    if cat_col in row_dict:
        row_dict[cat_col] = 1.0
        
    # Tenure Bucket
    tenure_col = f"Tenure Bucket_{comb['Tenure']}"
    if tenure_col in row_dict:
        row_dict[tenure_col] = 1.0
        
    # Agent Shift
    shift_col = f"Agent Shift_{comb['Shift']}"
    if shift_col in row_dict:
        row_dict[shift_col] = 1.0
        
    # 2. DataFrame
    df_input = pd.DataFrame([row_dict], columns=selected_features)
    print("DataFrame for prediction:")
    print(df_input.to_string())
    
    # 3. Scale
    scaled_input = scaler.transform(df_input)
    
    # 4. Predict
    pred_class = model.predict(scaled_input)[0]
    pred_proba = model.predict_proba(scaled_input)[0]
    
    print("Scaled Vector:")
    print(scaled_input)
    print(f"Prediction: {pred_class} ({'Satisfied (CSAT 4-5)' if pred_class == 1 else 'Unsatisfied (CSAT 1-3)'})")
    print(f"Probabilities: {pred_proba} (Satisfaction: {pred_proba[1]*100:.2f}%)")
