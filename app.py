import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page layout and config
st.set_page_config(
    page_title="Flipkart CSAT Predictor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set styling for premium dark dashboard theme
st.markdown("""
<style>
    /* Global styles */
    .reportview-container {
        background-color: #0f172a;
    }
    
    /* Header Container styling */
    .header-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.05rem;
    }
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Result card styling */
    .card-satisfied {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        border: 2px solid #10b981;
        padding: 1.75rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.2);
    }
    .card-unsatisfied {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        border: 2px solid #ef4444;
        padding: 1.75rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.2);
    }
    .card-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
    }
    .card-text {
        font-size: 1.1rem;
        opacity: 0.95;
        margin-top: 0.5rem;
    }
    .card-probability {
        font-size: 2.2rem;
        font-weight: 900;
        margin-top: 0.5rem;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Data & Model Loading -----------------
MODEL_PATH = "flipkart_csat_model.pkl"
CSV_PATH = "Customer_support_data.csv"

@st.cache_resource
def load_model_pipeline():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model pickle file not found at: {MODEL_PATH}")
    artifacts = joblib.load(MODEL_PATH)
    return artifacts

@st.cache_data
def load_historical_data():
    if not os.path.exists(CSV_PATH):
        return None
    # Load only necessary columns for the analytics tab to save memory and load faster
    cols = ['channel_name', 'category', 'Tenure Bucket', 'Agent Shift', 'CSAT Score', 'Item_price']
    df = pd.read_csv(CSV_PATH, usecols=cols)
    return df

# Initialize model components inside a try-except
try:
    model_pipeline = load_model_pipeline()
    model = model_pipeline['model']
    scaler = model_pipeline['scaler']
    selected_features = model_pipeline['selected_features']
    model_loaded = True
except Exception as e:
    st.error(f"🔴 **Error loading model pipeline**: {e}")
    model_loaded = False

# ----------------- UI Layout -----------------
st.markdown("""
<div class="header-box">
    <h1 class="header-title">🛍️ Flipkart CSAT Predictor</h1>
    <div class="header-subtitle">Predict customer satisfaction scores and analyze historical customer support agent metrics.</div>
</div>
""", unsafe_allow_html=True)

# Define Tabs
tab1, tab2 = st.tabs(["🔮 CSAT Prediction Engine", "📊 Historical Insights & Analytics"])

with tab1:
    if model_loaded:
        st.write("Enter the customer support interaction parameters to predict whether the customer will be satisfied.")
        
        # Predefined options from CSV
        channels = ['Inbound', 'Outcall', 'Email']
        categories = [
            'Cancellation', 'Order Related', 'Others', 'Payments related', 
            'Product Queries', 'Returns', 'Refund Related', 'Shopzilla Related', 
            'Feedback', 'Offers & Cashback', 'Onboarding related', 'App/website'
        ]
        tenure_buckets = ['0-30', '31-60', '61-90', '>90', 'On Job Training']
        agent_shifts = ['Morning', 'Afternoon', 'Evening', 'Night', 'Split']
        
        # Creating a form layout with 2 columns
        col1, col2 = st.columns([1, 1.2], gap="large")
        
        with col1:
            st.subheader("📋 Interaction Details")
            with st.form("prediction_form"):
                item_price = st.number_input(
                    "Item Price (INR)",
                    min_value=0.0,
                    max_value=20000.0,
                    value=500.0,
                    step=50.0,
                    help="Retail price of the product associated with the interaction."
                )
                
                channel = st.selectbox(
                    "Support Channel",
                    options=channels,
                    help="Communication channel used for the ticket."
                )
                
                category = st.selectbox(
                    "Interaction Category",
                    options=categories,
                    help="Primary reason or area of the customer ticket."
                )
                
                tenure = st.selectbox(
                    "Agent Tenure Bucket (Days)",
                    options=tenure_buckets,
                    help="The agent's experience/tenure bucket."
                )
                
                shift = st.selectbox(
                    "Agent Shift",
                    options=agent_shifts,
                    help="The work shift of the customer support agent."
                )
                
                submit_btn = st.form_submit_button("Predict Satisfaction")
        
        with col2:
            st.subheader("🎯 Prediction Output")
            if submit_btn:
                try:
                    # 1. Re-construct the dictionary matching 'selected_features'
                    row_dict = {feat: 0.0 for feat in selected_features}
                    
                    # Apply log1p transformation to Item_price as identified in notebook code
                    # (Item_price was transformed if response_time_minutes was missing from features)
                    row_dict['Item_price'] = np.log1p(float(item_price))
                    
                    # Map Channel dummy column
                    if channel == 'Inbound':
                        row_dict['channel_name_Inbound'] = 1.0
                        
                    # Map Category dummy column
                    cat_col = f"category_{category}"
                    if cat_col in row_dict:
                        row_dict[cat_col] = 1.0
                        
                    # Map Tenure Bucket dummy column
                    tenure_col = f"Tenure Bucket_{tenure}"
                    if tenure_col in row_dict:
                        row_dict[tenure_col] = 1.0
                        
                    # Map Agent Shift dummy column
                    shift_col = f"Agent Shift_{shift}"
                    if shift_col in row_dict:
                        row_dict[shift_col] = 1.0
                        
                    # 2. Build single-row DataFrame maintaining the strict feature order
                    df_input = pd.DataFrame([row_dict], columns=selected_features)
                    
                    # 3. Apply StandardScaler
                    scaled_input = scaler.transform(df_input)
                    
                    # 4. Model Prediction
                    pred_class = model.predict(scaled_input)[0]
                    pred_proba = model.predict_proba(scaled_input)[0]
                    prob_satisfied = pred_proba[1]  # Prob of class 1 (Satisfied)
                    
                    # Print exact DataFrame info for debugging / validation requests
                    print("\n[Streamlit Verification Output]")
                    print("Input Row DataFrame:")
                    print(df_input.to_string())
                    print("Scaled Feature Array:", scaled_input)
                    print(f"Predicted Class: {pred_class}, Probabilities: {pred_proba}")
                    
                    # 5. Display Custom Premium Result Cards
                    if pred_class == 1:
                        st.markdown(f"""
                        <div class="card-satisfied">
                            <div class="card-title">✅ Predicted: Satisfied (CSAT 4-5)</div>
                            <div class="card-text">The model predicts that this interaction will result in high customer satisfaction.</div>
                            <div class="card-probability">{prob_satisfied*100:.1f}%</div>
                            <div style="opacity:0.9; font-size:0.9rem;">Satisfaction Probability</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(float(prob_satisfied))
                    else:
                        st.markdown(f"""
                        <div class="card-unsatisfied">
                            <div class="card-title">⚠️ Predicted: Unsatisfied (CSAT 1-3)</div>
                            <div class="card-text">The model predicts that this interaction might result in low customer satisfaction.</div>
                            <div class="card-probability">{(1 - prob_satisfied)*100:.1f}%</div>
                            <div style="opacity:0.9; font-size:0.9rem;">Dissatisfaction Probability (Satisfaction: {prob_satisfied*100:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(float(prob_satisfied))
                        
                    # 6. Debug / Details Expander
                    with st.expander("🛠️ Raw Feature Vector Sent to Model"):
                        st.write("Here is the one-hot encoded and log-transformed input vector before scaling:")
                        st.dataframe(df_input)
                        st.write("Here is the final scaled feature vector passed to the GradientBoostingClassifier:")
                        st.dataframe(pd.DataFrame(scaled_input, columns=selected_features))
                        
                except Exception as ex:
                    st.error(f"🔴 **Prediction error occurred**: {ex}")
            else:
                st.info("👈 Enter details on the left and click **Predict Satisfaction** to run inference.")
    else:
        st.warning("⚠️ Prediction Engine is unavailable. Please resolve the model loading error.")

with tab2:
    st.subheader("📈 Historical Customer Support Insights")
    st.write("This analytics dashboard provides context on historical customer support performance from `Customer_support_data.csv`.")
    
    historical_df = load_historical_data()
    
    if historical_df is not None:
        # Create metric summary cards
        total_tickets = len(historical_df)
        avg_csat = historical_df['CSAT Score'].mean()
        
        # Satisfaction flag: CSAT >= 4
        historical_df['Satisfied'] = historical_df['CSAT Score'].isin([4, 5])
        overall_satisfaction = historical_df['Satisfied'].mean() * 100
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Total Interactions Analyzed", f"{total_tickets:,}")
        with m_col2:
            st.metric("Average CSAT Score", f"{avg_csat:.2f} / 5.0")
        with m_col3:
            st.metric("Overall Satisfaction Rate", f"{overall_satisfaction:.1f}%")
            
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        # Charts section
        c_col1, c_col2 = st.columns(2)
        
        with c_col1:
            st.subheader("Distribution of CSAT Scores")
            fig, ax = plt.subplots(figsize=(6, 4))
            # Custom palette: Reds for 1-3, Greens for 4-5
            palette = ['#e11d48' if score in [1, 2, 3] else '#10b981' for score in [1, 2, 3, 4, 5]]
            
            score_counts = historical_df['CSAT Score'].value_counts().sort_index()
            sns.barplot(x=score_counts.index, y=score_counts.values, palette=palette, ax=ax)
            ax.set_xlabel("CSAT Score (1 to 5)")
            ax.set_ylabel("Interaction Count")
            ax.set_title("CSAT Score Frequency (Green = Satisfied, Red = Unsatisfied)")
            st.pyplot(fig)
            
        with c_col2:
            st.subheader("Satisfaction Rate by Channel")
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Group by channel and compute satisfaction percentage
            channel_sat = historical_df.groupby('channel_name')['Satisfied'].mean().reset_index()
            channel_sat['Satisfaction Rate (%)'] = channel_sat['Satisfied'] * 100
            
            sns.barplot(
                x='channel_name', 
                y='Satisfaction Rate (%)', 
                data=channel_sat, 
                palette='Blues_r', 
                ax=ax
            )
            ax.set_ylim(0, 100)
            ax.set_ylabel("Satisfaction Rate (%)")
            ax.set_xlabel("Channel")
            ax.set_title("Percent of Satisfied Interactions (CSAT 4-5) by Channel")
            for p in ax.patches:
                ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() - 8),
                            ha='center', va='center', color='white', fontweight='bold', xytext=(0, 0), textcoords='offset points')
            st.pyplot(fig)
            
        st.markdown("<hr/>", unsafe_allow_html=True)
        
        c_col3, c_col4 = st.columns(2)
        with c_col3:
            st.subheader("Satisfaction Rate by Agent Experience (Tenure)")
            fig, ax = plt.subplots(figsize=(6, 4))
            
            tenure_sat = historical_df.groupby('Tenure Bucket')['Satisfied'].mean().reset_index()
            tenure_sat['Satisfaction Rate (%)'] = tenure_sat['Satisfied'] * 100
            
            # Sort chronologically for better reading
            tenure_order = ['0-30', '31-60', '61-90', '>90', 'On Job Training']
            tenure_sat['Tenure Bucket'] = pd.Categorical(tenure_sat['Tenure Bucket'], categories=tenure_order, ordered=True)
            tenure_sat = tenure_sat.sort_values('Tenure Bucket')
            
            sns.barplot(
                x='Tenure Bucket', 
                y='Satisfaction Rate (%)', 
                data=tenure_sat, 
                palette='Purples_r', 
                ax=ax
            )
            ax.set_ylim(0, 100)
            ax.set_ylabel("Satisfaction Rate (%)")
            ax.set_xlabel("Tenure Bucket")
            ax.set_title("Satisfaction Rate by Agent Tenure Bucket")
            st.pyplot(fig)
            
        with c_col4:
            st.subheader("Distribution of Support Channels")
            fig, ax = plt.subplots(figsize=(6, 4))
            channel_counts = historical_df['channel_name'].value_counts()
            ax.pie(channel_counts.values, labels=channel_counts.index, autopct='%1.1f%%', colors=['#3b82f6', '#10b981', '#f59e0b'], startangle=140)
            ax.set_title("Proportion of Interactions Across Support Channels")
            st.pyplot(fig)
    else:
        st.warning("⚠️ `Customer_support_data.csv` was not found. Please place it in the same directory to load summary charts.")
