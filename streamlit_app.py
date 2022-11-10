import streamlit as st
import pandas as pd
import numpy as np
from pydantic import BaseModel
import requests
from pydantic import parse_obj_as

class IncomeInput(BaseModel):
    Age: int
    Workclass: str
    Education: str 
    MaritalStatus: str
    Occupation: str
    Relationship: str
    Race: str 
    Sex: str 
    CapitalGain: int 
    CapitalLoss: int
    HoursPerWeek: int 
    NativeCountry: str

DATASET_PATH = "Data/adult.data"
#ENDPOINT = "http://127.0.0.1:3000/predict" #for local bento version
ENDPOINT = "http://adult-xgboost-app.westeurope.azurecontainer.io:3000/predict"

def main():
    @st.cache(persist=True)
    def load_dataset() -> pd.DataFrame:
        cols = ['Age', 'Workclass', 'Fnlwgt', 'Education','EducationNum', 'MaritalStatus', 'Occupation', 'Relationship', 'Race',
        'Sex', 'CapitalGain', 'CapitalLoss', 'HoursPerWeek', 'NativeCountry', 'Income'] 
        df = pd.read_csv("Data/adult.data", delimiter=",", header=None, names=cols, index_col=False)
        categorical_cols = df.select_dtypes(include=['O']).columns
        df[categorical_cols] = df[categorical_cols].apply(lambda x: x.str.strip())
        df[df == '?'] = np.nan
        return df


    def user_input_features() -> dict:
        Age = st.sidebar.number_input("Age", min_value=adult.Age.min(), max_value=adult.Age.max())
        Workclass = st.sidebar.selectbox("Workclass", options = adult.Workclass.dropna().unique())
        Education = st.sidebar.selectbox("Education", options = adult.Education.unique())
        MaritalStatus = st.sidebar.selectbox("Marital Status", options = adult.MaritalStatus.unique())
        Occupation = st.sidebar.selectbox("What is you Occupation?",options= adult.Occupation.dropna().unique())
        Relationship = st.sidebar.selectbox("Current Relationship status", options = adult.Relationship.unique())
        Race = st.sidebar.selectbox("Race", options = adult.Race.unique())
        Sex = st.sidebar.selectbox("Sex", options = adult.Sex.unique())
        CapitalGain = st.sidebar.number_input("What is your Capital Gain?", min_value=adult.CapitalGain.min(), max_value=adult.CapitalGain.max())
        CapitalLoss = st.sidebar.number_input("What is your Capital Loss?", min_value=adult.CapitalLoss.min(), max_value=adult.CapitalLoss.max())
        HoursPerWeek = st.sidebar.number_input("How many hours per week do you work?",min_value=adult.HoursPerWeek.min(), max_value=adult.HoursPerWeek.max())
        NativeCountry = st.sidebar.selectbox("NativeCountry", options=adult.NativeCountry.dropna().unique())
        
        data = {
            "Age": Age,
            "Workclass": Workclass,
            "Education": Education,
            "MaritalStatus": MaritalStatus,
            "Occupation": Occupation,
            "Relationship": Relationship,
            "Race": Race,
            "Sex": Sex,
            "CapitalGain": CapitalGain,
            "CapitalLoss": CapitalLoss,
            "HoursPerWeek": HoursPerWeek,
            "NativeCountry": NativeCountry
            }

        return data


    st.set_page_config(
        page_title="Income Prediction App",
        page_icon="images/income.jpg"
    )

    st.title("Income Prediction")
    st.subheader("Are you wondering if you earn over or under 50K dollars a year? "
                 "This app will predict that for you!")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("images/xgboost_robot.png",
                 caption="I'll help you predict your Income! - Mr. XGBOOST \U0001F609",
                 width=150)
        m = st.markdown("""
                        <style>
                        div.stButton > button:first-child {
                            background-color: #0099ff;
                            color:#ffffff;
                        }
                        div.stButton > button:hover {
                            background-color: #00ff00;
                            color:#000000;
                            }
                        </style>""", unsafe_allow_html=True)
        submit = st.button("Predict")
    with col2:
        st.markdown("""
        Did you know that machine learning models can help you predict
        whether your income is above or below 50K pretty accurately? In 
        this app, you can predict your income bracket (>50K/<50K) in seconds!
        
        Here, an XBoost (extreme gradient boosted decision tree) model
        was constructed using survey data of over 45k US residents from the year 1994.
        This application is based on it because it has proven to be a very good predictor
        for the income bracket.
        
        To predict your income bracket, simply follow the steps bellow:
        1. Enter the parameters that best describe you;
        2. Press the "Predict" button and wait for the result.
            

        **Author: Mohammed Arebi ([GitHub](https://github.com/arebimohammed))**
        
        You can see the steps of building the model, evaluating it, and cleaning the data itself
        on my GitHub repo [here](). 
        """)

    adult = load_dataset()

    st.sidebar.title("Survey")
    st.sidebar.image("images/income2.jpg", width=100)

    data = user_input_features()
    X = parse_obj_as(IncomeInput,data)

    if submit:
        response = requests.post(ENDPOINT, headers={"content-type": "application/json"}, data=X.json())
        prediction = response.text.strip("[").strip("]")
        
        if prediction == '0':
            st.markdown(f"**The model predicted that you income is"
                        f" below 50K."
                        f" Don't worry you're on your way!**")
            st.image("images/inc_notgood.jpg")
        else:
            st.markdown(f"**he model predicted that you income is"
                        f" above 50K."
                        f" Congradulations! Keep up the great work.**")
            st.image("images/inc_good.jpg")


if __name__ == "__main__":
    main()