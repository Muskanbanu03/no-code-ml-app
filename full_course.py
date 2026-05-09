import streamlit as st

"""st.title("Hello Streamlit")
st.write("This is my first streamlit app!")

st.title("Student Information Form")
st.subheader("Enter your information below")

name=st.text_input("Name")
age=st.slider("Age",0,50,1)
gender=st.radio("Gender",["Male","Female","Other"])
skills=st.selectbox("Skills",["Coding","Designing","Editing"])
st.write("Knowledge:")
python=st.checkbox("Python-model_building")
sql=st.checkbox("SQL-database")
tablue=st.checkbox("Tablue-dashboard")

if st.button("Submit"):
    st.subheader("Your Profile")
    st.write(f"Name: {name}")
    st.write(f"Age: {age}")
    st.write(f"Gender: {gender}")
    st.write(f"Skills: {skills}")
    if python:
        st.write(f"Knowledge: Python-Model_building")
    if sql:
        st.write(f"Knowledge: SQL-database")
    if tablue:
        st.write(f'Knowledge: Tablue-dashboard')"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
file = st.file_uploader("Upload your file here")

if file:
    df=pd.read_csv(file)
    st.write("**Top 5 rows:**", df.head())
    st.write("**Bottom 5 row:** ", df.tail())
    st.write("**Statistical summary:** ",df.describe())
    st.write("**Data types**", df.info())
    st.write("**Checking Duplicates**", df.duplicated().sum())
    st.write("**Checking Missing Data**", df.isnull().sum())

    st.subheader("Numerical columns distribution")
    num_cols=df.select_dtypes(include=['int','float']).columns
    #st.plotly_chart(sns.heatmap(df[num_cols]))

    target=st.selectbox("Select target column",df.columns)
    x=df.drop(target, axis=1)
    y=df[target]

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.3,random_state=1)

    """from sklearn.ensemble import RandomForestClassifier
    model=RandomForestClassifier()
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    acc=accuracy_score(y_test,pred)
    st.success(f'Model trained with accuracy: {acc:.2f}')"""

    st.download_button('Download CSV', df.to_csv(), 'data.csv')
    