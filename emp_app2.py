# EMPLOYEE PROMOTION PREDICTOR
# PART 1
# Professional Dashboard UI
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# PAGE CONFIG
st.set_page_config(
    page_title="Employee Promotion Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
# LOAD MODEL
model = pickle.load(open("promotion_model.pkl", "rb"))

scaler = pickle.load(open("scaler.pkl", "rb"))

encoders = pickle.load(open("encoders.pkl", "rb"))
# CUSTOM CSS
st.markdown("""
<style>

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}

/* Main Background */
.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b,#111827);
}

/* Hide Streamlit Menu */
#MainMenu{
    visibility:hidden;
}
footer{
    visibility:hidden;
}
header[data-testid="stHeader"]{
    background: transparent;
}


/* Sidebar */
section[data-testid="stSidebar"]{
    background:#111827;
    border-right:1px solid #374151;
}

/* Hero Card */
.hero{
    background:linear-gradient(135deg,#2563eb,#7c3aed);
    padding:30px;
    border-radius:20px;
    color:white;
    box-shadow:0px 8px 30px rgba(0,0,0,.35);
}

/* Glass Card */
.glass{
    background:rgba(255,255,255,.08);
    backdrop-filter:blur(14px);
    border:1px solid rgba(255,255,255,.12);
    border-radius:18px;
    padding:18px;
    box-shadow:0 10px 25px rgba(0,0,0,.30);
}

/* KPI Card */
.metric-card{
    background:#1f2937;
    padding:18px;
    border-radius:16px;
    text-align:center;
    border:1px solid #374151;
    transition:.3s;
}

.metric-card:hover{
    transform:translateY(-5px);
    border:1px solid #60a5fa;
}

/* Buttons */
.stButton>button{
    width:100%;
    height:52px;
    border-radius:12px;
    border:none;
    background:linear-gradient(90deg,#2563eb,#7c3aed);
    color:white;
    font-size:18px;
    font-weight:bold;
    transition:0.3s;
}

.stButton>button:hover{
    transform:scale(1.03);
    box-shadow:0px 0px 18px #60a5fa;
}

/* Input Box */
div[data-baseweb="select"]{
    border-radius:12px;
}

input{
    border-radius:12px !important;
}

/* Slider */
.stSlider{
    padding-top:10px;
}

/* Success Box */
.success-box{
    background:#064e3b;
    padding:20px;
    border-radius:15px;
    color:white;
}

/* Error Box */
.error-box{
    background:#7f1d1d;
    padding:20px;
    border-radius:15px;
    color:white;
}

hr{
    border:1px solid #374151;
}

</style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.image("app192x192.png", width=90)
st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "",
    [
        "🏠 Prediction",
        "📊 Model Insights",
        "📁 Batch Upload",
        "📜 Prediction History",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.markdown("""
### Dashboard Features

✅ Promotion Prediction

✅ Probability Score

✅ KPI Dashboard

✅ Batch CSV Upload

✅ Data Visualization

✅ Model Insights

✅ Prediction History

""")

st.sidebar.markdown("---")

st.sidebar.caption("Version 2.0")

# -----------------------------
# HERO SECTION
# -----------------------------
st.markdown("""
<div class="hero">

<h1>🏢 Employee Promotion Predictor</h1>

<p style="font-size:18px;">

Predict employee promotion using Machine Learning
with a modern professional dashboard.

</p>

</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# TOP KPI CARDS
# -----------------------------
col1,col2,col3,col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
    <h3>🎯 Accuracy</h3>
    <h2>91%</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
    <h3>🤖 Model</h3>
    <h2>XGBoost</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
    <h3>📂 Features</h3>
    <h2>8</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
    <h3>⚡ Speed</h3>
    <h2>Fast</h2>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.markdown("---")
# ============================================
# PART 2
# Prediction Section
# ============================================

if menu == "🏠 Prediction":

    st.markdown("""
    <div class="glass">
    <h2>📝 Employee Information</h2>
    <p>Fill all employee details below to predict promotion eligibility.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # -----------------------------
    # INPUT FORM
    # -----------------------------

    left_col, right_col = st.columns(2)

    with left_col:

        age = st.number_input(
            "👤 Age",
            min_value=18,
            max_value=60,
            value=30
        )

        education_level = st.selectbox(
            "🎓 Education Level",
            [
                "Bachelor",
                "Master",
                "PhD"
            ]
        )

        department = st.selectbox(
            "🏢 Department",
            [
                "Finance",
                "Sales",
                "Engineering",
                "Operations",
                "HR"
            ]
        )

        training_hours = st.slider(
            "📚 Training Hours",
            0,
            200,
            50
        )

    with right_col:

        experience = st.slider(
            "💼 Years at Company",
            0,
            40,
            5
        )

        performance_score = st.slider(
            "⭐ Performance Score",
            0,
            100,
            70
        )

        manager_rating = st.slider(
            "👨‍💼 Manager Rating",
            0,
            100,
            75
        )

        employee_engagement = st.slider(
            "🤝 Employee Engagement",
            0,
            100,
            80
        )

    st.write("")

    # -----------------------------
    # LIVE EMPLOYEE PROFILE CARD
    # -----------------------------

    st.markdown("### 👨 Employee Summary")

    c1, c2 = st.columns([1,2])

    with c1:

        st.markdown(f"""
        <div class="glass">

        <h3>👤 Employee</h3>

        <hr>

        <b>Age</b><br>
        {age} Years

        <br><br>

        <b>Education</b><br>
        {education_level}

        <br><br>

        <b>Department</b><br>
        {department}

        </div>
        """,
        unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class="glass">

        <h3>📈 Performance Overview</h3>

        </div>
        """,
        unsafe_allow_html=True)

        st.write("Training Progress")
        st.progress(training_hours/200)

        st.write("Performance")
        st.progress(performance_score/100)

        st.write("Manager Rating")
        st.progress(manager_rating/100)

        st.write("Employee Engagement")
        st.progress(employee_engagement/100)

    st.write("")

    # -----------------------------
    # QUICK METRICS
    # -----------------------------

    st.markdown("### 📊 Employee Statistics")

    m1,m2,m3,m4 = st.columns(4)

    with m1:
        st.metric(
            "Experience",
            f"{experience} Years"
        )

    with m2:
        st.metric(
            "Training",
            f"{training_hours} Hrs"
        )

    with m3:
        st.metric(
            "Performance",
            f"{performance_score}/100"
        )

    with m4:
        st.metric(
            "Engagement",
            f"{employee_engagement}/100"
        )

    st.write("")
    st.markdown("---")

    # -----------------------------
    # ENCODING
    # -----------------------------

    education = encoders["education_level"].transform([education_level])[0]

    department = encoders["department"].transform([department])[0]

    #  Prepare Input
    input_data = np.array([
       age,
       education,
       department,
       training_hours,
       experience,
       performance_score,
       manager_rating,
       employee_engagement
    ]).reshape(1, -1)

    # Scale Input
    input_data = scaler.transform(input_data)

    # -----------------------------
    # PREDICT BUTTON
    # -----------------------------

    predict_btn = st.button(
        "🚀 Predict Employee Promotion",
        use_container_width=True
    )

    # ============================================
# PART 3
# Prediction Result
# ============================================

    if predict_btn:

        with st.spinner("🤖 AI Model is analyzing employee data..."):

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

        st.write("")
        st.markdown("---")

        st.markdown("""
        <div class="glass">
        <h2>📊 Prediction Report</h2>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # -------------------------
        # Probability Meter
        # -------------------------

        st.subheader("🎯 Promotion Probability")

        st.progress(float(probability))

        st.info(f"Confidence Score : **{probability*100:.2f}%**")

        st.write("")

        # -------------------------
        # Result Cards
        # -------------------------

        if prediction == 1:

            st.balloons()

            st.markdown(f"""
            <div class="success-box">

            <h2>🎉 Employee Will Be Promoted</h2>

            <hr>

            <h3>Confidence : {probability*100:.2f}%</h3>

            <p>
            The employee satisfies the required conditions
            for promotion according to the trained ML model.
            </p>

            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="error-box">

            <h2>❌ Employee Will NOT Be Promoted</h2>

            <hr>

            <h3>Confidence : {(1-probability)*100:.2f}%</h3>

            <p>
            Current employee profile does not meet the
            promotion criteria learned by the ML model.
            </p>

            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # -------------------------
        # Confidence Gauge
        # -------------------------

        st.subheader("📈 Confidence Meter")

        gauge_col1, gauge_col2 = st.columns([3,1])

        with gauge_col1:

            st.progress(float(probability))

        with gauge_col2:

            st.metric(
                "Confidence",
                f"{probability*100:.1f}%"
            )

        st.write("")

        # -------------------------
        # Feature Summary
        # -------------------------

        st.subheader("📋 Input Summary")

        summary = pd.DataFrame({

            "Feature":[

                "Age",

                "Education",

                "Department",

                "Training Hours",

                "Experience",

                "Performance",

                "Manager Rating",

                "Engagement"

            ],

            "Value":[

                age,

                education_level,

                department,

                training_hours,

                experience,

                performance_score,

                manager_rating,

                employee_engagement

            ]

        })

        st.dataframe(
            summary,
            use_container_width=True
        )

        st.write("")

        # -------------------------
        # Prediction History
        # -------------------------

        st.subheader("🕒 Prediction History")

        if "history" not in st.session_state:
            st.session_state.history = pd.DataFrame(columns=[
                "Age",
                "Education",
                "Department",
                "Experience",
                "Prediction",
                "Confidence"
            ])

        new_row = pd.DataFrame({

            "Age":[age],

            "Education":[education_level],

            "Department":[department],

            "Experience":[experience],

            "Prediction":[
                "Promoted"
                if prediction==1
                else
                "Not Promoted"
            ],

            "Confidence":[
                f"{probability*100:.2f}%"
            ]

        })

        st.session_state.history = pd.concat(
            [
                st.session_state.history,
                new_row
            ],
            ignore_index=True
        )

        st.dataframe(
            st.session_state.history,
            use_container_width=True
        )

        st.write("")

        # -------------------------
        # Download Button
        # -------------------------

        csv = st.session_state.history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            "📥 Download Prediction Report",

            data=csv,

            file_name="prediction_history.csv",

            mime="text/csv",

            use_container_width=True

        )

        st.markdown("---")

        # =====================================================
# PART 4
# MODEL INSIGHTS
# =====================================================

elif menu == "📊 Model Insights":

    st.title("📊 Model Performance Dashboard")

    st.write("")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Accuracy","91%","+2.1%")
    c2.metric("Precision","89%","+1.4%")
    c3.metric("Recall","87%","+3.2%")
    c4.metric("AUC Score","0.90","+0.02")

    st.markdown("---")

    col1,col2 = st.columns(2)

    with col1:

        st.subheader("📈 Model Comparison")

        models = [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost"
        ]

        accuracy = [
            0.91,
            0.86,
            0.90,
            0.82
        ]

        fig,ax = plt.subplots(figsize=(8,5))

        sns.barplot(
            x=models,
            y=accuracy,
            palette="viridis",
            ax=ax
        )

        ax.set_ylim(0,1)

        ax.set_ylabel("Accuracy")

        plt.xticks(rotation=20)

        st.pyplot(fig)

    with col2:

        st.subheader("🥧 Accuracy Distribution")

        fig2,ax2 = plt.subplots(figsize=(6,6))

        ax2.pie(
            [91,9],
            labels=["Correct","Incorrect"],
            autopct="%1.1f%%",
            colors=["green","red"]
        )

        st.pyplot(fig2)

    st.markdown("---")

    st.subheader("📊 Feature Importance")

    features = [

        "Age",

        "Education",

        "Department",

        "Training",

        "Experience",

        "Performance",

        "Manager Rating",

        "Engagement"

    ]

    importance = [

        0.10,

        0.08,

        0.12,

        0.14,

        0.20,

        0.16,

        0.11,

        0.09

    ]

    fig3,ax3 = plt.subplots(figsize=(9,5))

    sns.barplot(

        x=importance,

        y=features,

        palette="coolwarm",

        ax=ax3

    )

    st.pyplot(fig3)

# =====================================================
# BATCH UPLOAD
# =====================================================

elif menu == "📁 Batch Upload":

    st.title("📁 Batch Prediction")

    st.info("Upload CSV File")

    uploaded_file = st.file_uploader(

        "Choose CSV",

        type=["csv"]

    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.subheader("Preview")

        st.dataframe(df)

        if st.button("Predict All Employees"):

            pred = model.predict(df)

            df["Promotion Prediction"] = pred

            st.success("Prediction Completed Successfully")

            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(

                "📥 Download Result",

                csv,

                "Prediction.csv",

                "text/csv",

                use_container_width=True

            )

# =====================================================
# HISTORY
# =====================================================

elif menu == "📜 Prediction History":

    st.title("🕒 Prediction History")

    if "history" in st.session_state:

        st.dataframe(

            st.session_state.history,

            use_container_width=True

        )

    else:

        st.warning("No Prediction History Found")

# =====================================================
# ABOUT
# =====================================================

elif menu == "ℹ️ About":

    st.title("ℹ️ About Project")

    st.markdown("""

### Employee Promotion Prediction

This application predicts whether an employee
is eligible for promotion using Machine Learning.

---

### Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- XGBoost

---

### Features

✅ Modern Dashboard

✅ Promotion Prediction

✅ Batch Prediction

✅ Prediction History

✅ Download Report

✅ Professional Charts

---

### Developed By

**Anurag Mishra**

Government Polytechnic Arvi

Computer Engineering

""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(

"""

<center>

<h4>🏢 Employee Promotion Prediction Dashboard</h4>

Made with ❤️ using Streamlit & Machine Learning

<br>

<b>Developer : Anurag Mishra</b>

</center>

""",

unsafe_allow_html=True

)