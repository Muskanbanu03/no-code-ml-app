import streamlit as st 
import pandas as pd 

st.title("Chai Sales Dashboard")

file=st.file_uploader("Upload your file", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Data preview")
    st.dataframe(df)

if file:
    st.subheader("Summary stats")
    st.write(df.describe())

if file:
    gender=df['Gender'].unique()
    selected_gender=st.selectbox("Filter by Gender", gender)
    filtered_data=df[df['Gender']==selected_gender]
    st.dataframe(filtered_data)