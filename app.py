import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# App Title
# -----------------------------
st.set_page_config(page_title="Student Performance Prediction", layout="centered")

st.title("🎓 Student Performance Prediction")
st.write("Predict whether a student will **Pass or Fail** based on academic and personal factors.")

#model selection function
def model_selector(model: str, input_data):
    if model == "Decision Tree":
        with open('DecisionTree.pkl','br') as f:
            model = pickle.load(f)
        return model.predict(input_data)[0]
    else:
        with open('RandomForest.pkl','br') as f:
            model = pickle.load(f)
        return model.predict(input_data)[0]

# -----------------------------
# User Inputs
# -----------------------------
st.header("📥 Enter Student Details")
age = st.slider("Age of student",0,22,1)
age = st.slider("Age of student",0,22,1,value=17)
famsize = st.selectbox("Size of family(less than 3[LE3], greater than 3[GT3])",['LE3','GT3'],index=0)
Medu = st.selectbox("mother's education (0 - none,  1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)",[0,1,2,3,4],index=0)
Fedu = st.selectbox("Father's education (0 - none,  1 - primary education (4th grade), 2 - 5th to 9th grade, 3 - secondary education or 4 - higher education)",[0,1,2,3,4],index=0)
Mjob = st.selectbox("mother's job ('teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')",['teacher','health','services','at_home','other'],index=4)
guardian = st.selectbox("student's guardian ('mother', 'father' or other')",['mother','father','other'],index=2)
traveltime = st.selectbox("home to school travel time (1 - less than 15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - greater than 1 hour)",[1,2,3,4],index=2)
study_time = st.slider("weekly study time (1 - less than 2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - greater than 10 hours)",1,4,1,value=3)
failures = st.selectbox("past class failures", [0, 1, 2, 3],index=0)
schoolsub = st.selectbox("extra educational support (yes or no)",[1,0],index=0)
activities = st.selectbox("extra-curricular activities (yes or no)",[1,0],index=0)
higher = st.selectbox("wants to take higher education",[1,0],index=0)
internet = st.selectbox("Internet Access", [1,0],index=0)
romantic = st.selectbox("romantic relationship",[1,0],index=0)
famrel = st.select_slider("quality of family relationships (from 1 - very bad to 5 - excellent)",1,5,1,value=3)
freetime = st.select_slider("free time after school (from 1 - very low to 5 - very high)",1,5,1,value=3)
goout = st.select_slider("going out with friends (from 1 - very low to 5 - very high)",1,5,1,value=3)
Dalc = st.select_slider("workday alcohol consumption (from 1 - very low to 5 - very high)",1,5,1,value=3)
Walc = st.select_slider("weekend alcohol consumption (from 1 - very low to 5 - very high)",1,5,1,value=3)
health = st.select_slider("current health status (from 1 - very bad to 5 - very good)",1,5,1,value=3)
absences = st.slider("Number of Absences", 0, 93, 2,value= 10)
G1 = st.slider("first period grade (from 0 to 20)",0,20,1,value=13)
G2 = st.slider("second period grade (from 0 to 20)",0,20,1,value=13)
model_selection = st.selectbox("Select the model for Prediction",["Decision Tree","Random Forest"],index=0) 
# -----------------------------
# Encode Categorical Inputs
# -----------------------------

input_data = pd.DataFrame({
    "age": [age],
    "famsize": [famsize],
    "Medu": [Medu],
    "Fedu": [Fedu],
    "Mjob": [Mjob],
    "guardian": [guardian],
    "traveltime": [traveltime],
    "studytime": [study_time],
    "failures": [failures],
    "schoolsup": [schoolsub],
    "activities": [activities],
    "higher": [higher],
    "internet": [internet],
    "romantic": [romantic],
    "famrel": [famrel],
    "freetime": [freetime],
    "goout": [goout],
    "Dalc": [Dalc],
    "Walc": [Walc],
    "health": [health],
    "absences": [absences],
    "G1": [G1],
    "G2": [G2]
})

le = LabelEncoder()
for col in input_data.select_dtypes(include='object').columns:
    input_data[col] = le.fit_transform(input_data[col])



# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Result"):
    
    
    prediction = model_selector(model_selection,input_data)
    
    if prediction == 1:
        st.success("✅ Prediction: **PASS**")
    else:
        st.error("❌ Prediction: **FAIL**")


# -----------------------------
# Feature Importance Graph
# -----------------------------

df = pd.read_csv("student-por.csv",sep = ";")
df.head()

X = df.drop(['G3', 'school','sex', 'address', 'reason','Pstatus','paid','famsup','nursery'], axis=1)
y = df['G3']
importances = rf_pipeline.named_steps["model"].feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

feat_imp.plot(kind="bar", figsize=(10,5))
plt.title("Feature Importance - Random Forest Pipeline")
plt.show()


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("📊 ML Model: Decision Tree / Random Forest")
