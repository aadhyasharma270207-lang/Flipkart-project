# 🛍️ Flipkart CSAT Predictor

A modern, high-performance Streamlit web application that predicts customer satisfaction (CSAT) scores for Flipkart customer support interactions. The application uses a pre-trained Gradient Boosting Classifier model, performing preprocessing, log-transformation, scaling, and classification in real time. It also features a historical insights dashboard visualizing key performance indicators from support logs.

Live App URL: [Streamlit Community Cloud Link](https://flipkart-project-m6hmrzkx3mcawrpmnz4pbj.streamlit.app/)

---

## 🔮 Key Features

1. **CSAT Prediction Engine (Tab 1)**:
   - Input forms for item price, communication channel, interaction category, agent tenure bucket, and work shift.
   - Real-time preprocessing mapping inputs into one-hot dummy variables matching the model's exact expected column structure.
   - Pre-prediction log-transformation (`np.log1p`) on numeric features to align with the training distribution.
   - Color-coded indicator card showcasing the prediction class (Satisfied vs. Unsatisfied).
   - Satisfaction percentage gauge showing prediction probability.
   - Collapsible debug expander displaying the raw and standard-scaled feature vectors sent to the model.

2. **Historical Insights & Analytics (Tab 2)**:
   - Overall metric summaries (Total tickets, average CSAT, satisfaction rate).
   - Interactive Seaborn/Matplotlib charts depicting CSAT score distributions, satisfaction rate by support channel, satisfaction by agent experience, and proportion of interactions across channels.
   - Optimized with `st.cache_data` to load files lazily and avoid overhead during prediction tasks.

---

## 📂 Project Structure

- `app.py`: The main Streamlit web application.
- `flipkart_csat_model.pkl`: Pre-trained Gradient Boosting Classifier, StandardScaler, and selected features list.
- `Customer_support_data.csv`: Support log dataset used for historical analytics.
- `requirements.txt`: Python package dependencies.
- `verify_correctness.py`: Preprocessing and unpickling verification script.
- `run_test_predictions.py`: Script to verify specific test combinations locally.

---

## 🚀 How to Run Locally

### 1. Prerequisites
Make sure you have Python 3.9+ installed.

### 2. Install Dependencies
Install all package requirements:
```bash
pip install -r requirements.txt
```

### 3. Run the Web App
Start the local Streamlit server:
```bash
streamlit run app.py
```
Open [Streamlit App Link](https://flipkart-project-m6hmrzkx3mcawrpmnz4pbj.streamlit.app/) in your browser.

---

## 🌐 Deployment to Streamlit Cloud

To host this app on the web:
1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **"New app"** and select the repository: `aadhyasharma270207-lang/Flipkart-project`.
3. Set the Branch to `main` and Main file path to `app.py`.
4. Click **Deploy!**
