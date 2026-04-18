import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =======================================================================================================
# PAGE CONFIG 
st.set_page_config(page_title="AI Retention Copilot", layout="wide")

# =======================================================================================================
# HEADER 
st.markdown("""
    <h1 style='text-align: center; color: saddlebrown;'>
    AI Retention & Activation Copilot
    </h1>
    <p style='text-align: center; font-size:18px;'>
    Predict churn, explain risk, and drive smarter fintech decisions
    </p>
""", unsafe_allow_html=True)
st.markdown("---")

# =======================================================================================================
# FILE UPLOAD 
st.subheader("***Upload Customer Dataset***")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# =======================================================================================================
# MAIN SECTION
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("*File uploaded successfully*")
    st.subheader("Customer Data Overview")
    st.dataframe(df.head())
    st.subheader("Data Quality Summary")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        st.success("No missing values found.")
    else:
        st.warning("Missing values detected:")
        st.write(missing[missing > 0])

    # =======================================================================================================
    # PREPROCESSING 
    df.drop(columns=["customer_id"], inplace=True, errors='ignore')
    df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
    df['gender'] = df['gender'].fillna(0)
    if 'country' in df.columns:
        df = pd.get_dummies(df, columns=['country'], drop_first=True)

    # =======================================================================================================
    # FEATURE ENGINEERING
    df['balance_salary_ratio'] = df['balance'] / (df['estimated_salary'] + 1)
    df['tenure_age_ratio'] = df['tenure'] / (df['age'] + 1)
    df['activity_score'] = df['active_member'] * df['products_number']

    # =======================================================================================================
    # MODEL
    X = df.drop(columns=['churn'])
    y = df['churn']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_scaled, y)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    threshold = st.slider("Select Churn Threshold", 0.1, 0.9, 0.4)
    y_pred = (y_prob > threshold).astype(int)
    df['Churn_Probability'] = y_prob
    df['Prediction'] = y_pred
    st.markdown("---")
    st.write("Model trained using logistic regression with balanced class weights to handle churn imbalance.")

    # =======================================================================================================
    # PRIORITY ENGINE
    df['Priority_Score'] = df['Churn_Probability'] * df['balance']

    # =======================================================================================================
    # CHURN REASON 
    def churn_reason(row):
        reasons = []
        if row['balance'] == 0:
            reasons.append("Low balance")
        if row['active_member'] == 0:
            reasons.append("Inactive user")
        if row['products_number'] <= 1:
            reasons.append("Low engagement")
        return ", ".join(reasons) if reasons else "Stable"
    df['Churn_Reason'] = df.apply(churn_reason, axis=1)

    # =======================================================================================================
    # PRODUCT RECOMMENDATION 
    def recommend_product(row):
        if row['balance'] > 100000:
            return "Fixed Deposit"
        elif row['balance'] > 50000:
            return "Recurring Deposit"
        else:
            return "Savings / Engagement"
    df['Product_Recommendation'] = df.apply(recommend_product, axis=1)

    # =======================================================================================================
    # ACTION ENGINE 
    def recommend_action(row):
        if row['Churn_Probability'] > 0.7:
            return "High Risk - Offer Incentive"
        elif row['Churn_Probability'] > 0.4:
            return "Medium Risk - Engage"
        else:
            return "Low Risk - Monitor"
    df['Action'] = df.apply(recommend_action, axis=1)

    # =======================================================================================================
    # Business Metric 
    st.markdown("---")
    st.subheader("Business Metrics")
    churn_rate = df['Prediction'].mean()
    high_risk = df[df['Churn_Probability'] > 0.7].shape[0]
    revenue_risk = df[df['Prediction'] == 1]['balance'].sum()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Churn Rate", f"{churn_rate:.2%}")
    col2.metric("High Risk Users", high_risk)
    col3.metric("Total Customers", len(df))
    col4.metric("Revenue at Risk", f"₹{revenue_risk:,.0f}")

    # ======================================================================================================= 
    # Customer Segmentation
    st.markdown("---")
    st.subheader("Customer Segmentation")

    risk_filter = st.selectbox(
        "Select Risk Level",
        ["All", "High Risk", "Medium Risk", "Low Risk"]
    )

    if risk_filter == "High Risk":
        filtered_df = df[df['Churn_Probability'] > 0.7]
    elif risk_filter == "Medium Risk":
        filtered_df = df[(df['Churn_Probability'] > 0.4) & (df['Churn_Probability'] <= 0.7)]
    elif risk_filter == "Low Risk":
        filtered_df = df[df['Churn_Probability'] <= 0.4]
    else:
        filtered_df = df
        st.dataframe(filtered_df, width='stretch')

    # =======================================================================================================
    # CHARTS
    # # Customer Churn Breakdown chart
    st.markdown("---")
    st.subheader("Customer Churn Breakdown")
    df['Churn_Label'] = df['Prediction'].map({0: 'No Churn', 1: 'Churn'})
    churn_counts = df['Churn_Label'].value_counts().reset_index()
    churn_counts.columns = ['Churn', 'Count']
    fig = px.pie(
        churn_counts,
        names='Churn',
        values='Count',
        hole=0.5,
        color='Churn',
        color_discrete_map={
            'No Churn': "tan",   
            'Churn': 'saddlebrown'     
            }
            )
    fig.update_traces(
    textinfo='percent+label',
    textfont_size=14,
    marker=dict(line=dict(color='saddlebrown', width=1)) 
    )
    fig.update_layout(
        showlegend=True,
        title="Churn Distribution"
        )
    st.plotly_chart(fig, width='stretch')
    
    #Risk by Age Group Chart
    st.subheader("Churn Risk by Age Segment")
    df['age_group'] = pd.cut(df['age'], bins=[18,30,50,70,100],
                         labels=['Young','Mid','Senior','Old'])
    age_risk = df.groupby('age_group', observed=False)['Churn_Probability'].mean().reset_index()
    fig = px.bar(
    age_risk,
    x='age_group',
    y='Churn_Probability',
    color='age_group',
    text='Churn_Probability',
    color_discrete_sequence=[
        "wheat",  
        'burlywood',  
        'tan',  
        'saddlebrown'   
    ]
)
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig, width='stretch')
    
    # =======================================================================================================
    # TOP USERS
    st.markdown("---")
    st.subheader("High Priority Customers")
    top_users = df.sort_values(by='Priority_Score', ascending=False).head(10)
    st.dataframe(top_users, width='stretch')

    # =======================================================================================================
    # FINAL OUTPUT 
    st.markdown("---")
    st.subheader("Detailed Customer Risk View")
    st.dataframe(df.head(20), width='stretch')

    # ======================================================================================================= 
    # FINAL MESSAGE
    st.success("Analysis Completed Successfully!")
    st.info("Customers with low balance and inactivity are consistently showing higher churn probability."
    "Focusing on engagement strategies for these users can reduce churn.")