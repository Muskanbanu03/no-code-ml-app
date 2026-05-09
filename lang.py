"""import streamlit as st

st.title('programming language selection')
st.subheader('Language')
st.write('Welcome to your first interactive app')
st.write('Choose your fav application, Variety of Apps:')

app=st.selectbox("Your fav app:", ['Python','Java','Java Script','HTML'])
st.write(f'You chose {app}, Excellent choice')

st.success("Your app is ready to code")

import streamlit as st

st.title("🧑‍💼 Simple Profile Generator")
st.subheader("Enter Your Information")
st.text("Fill in the details below to generate your profile card.")

name=st.text_input("Enter your name:")
age=st.text_input("Enter your age")

Profession=st.radio("Select your Profession",["Data Scientist","Doctor","Software Engineer","Hardware Engineer","Other"])
if Profession=="Other":
    prof_input=st.text_input("Enter your profession:")

hobby=st.selectbox("Choose your hobby",["Coding", "Painting","Paper crafts", "Reading books", "Cooking"])

if st.button("Generate Profile"):

    st.subheader("📌 Your Profile Summary")
    st.write(f"Name: {name}")
    st.write(f"Age: {age}")
    if Profession =="Other":
        st.write(f"Profession: {prof_input}")
    else:
        st.write(f"Profession: {Profession}")
    st.write(f"Hobby: {hobby}")
    
    st.success("Your Profile generated successfully!")

import streamlit as st

st.title("User Information Collector")

name = st.text_input("Name")
age=st.slider("Age",0,40,25)

gender=st.radio("Gender",["Male","Female","Other"])

st.subheader("Programming Languages")
lang_python=st.checkbox("Python")
lang_js=st.checkbox("JavaScript")
lang_cpp=st.checkbox("C++")
languages = st.multiselect(
    "Programming Languages",
    ["Python", "JavaScript", "C++"]
)


exp=st.selectbox("Years of Experience",[1,2,3,4,5,6,7,8,9,10])
bio=st.text_input("Bio")

if st.button("Submit"):
    st.success("Successfully submitted your profile")
    st.write('### Your Details:')
    st.write("**Name:**",name)
    st.write("**Age:**",age)
    st.write("**Gender:**",gender)
    selected_lang=[]
    if lang_python:
        selected_lang.append("Python")
    if lang_js:
        selected_langs.append("Javascipt")
    if lang_cpp:
        selected_lang.append("C++")
    st.write(selected_lang)
    st.write("**Experience:**",exp)
    st.write("**Bio:**", bio)


import streamlit as st

st.sidebar.title("Navigation")
st.sidebar.text("Use the sidebar to explore options 👈")
st.sidebar.image("https://cdn.analyticsvidhya.com/wp-content/uploads/2020/12/spark21.png")

st.header("🌟 User Information Dashboard")

col1, col2= st.columns(2)

with col1:
    st.subheader("👤 Personal Info")
    name=st.text_input("Name")
    age=st.slider("Age",1,100,25)
    gender=st.radio("Gender",["Male","Female","Other"])
    dob=st.date_input("Date of birth")

with col2:
    st.subheader("📚 Professional Info")
    profession=st.selectbox("Select your Profession",["Data Scientist","Doctor","Software Engineer","Teacher","Other"])
    experience=st.number_input("Years of Experience",0,50,1)

st.subheader("💻 Programming Skills")
python=st.checkbox("Python")
javascript=st.checkbox("JavaScript")
c=st.checkbox("C++")

with st.expander("Write a short bio about yourself"):
    bio=st.text_input("Bio")

if st.button("Submit"):
    st.write(f"**Name:** {name}")
    st.write(f"**Age:** {age}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Date Of Birth:** {dob}")
    st.write(f'**Profession:** {profession}')
    st.write(f'**Experience:** {experience}')
    if python:
        st.write(f"**Language Know:** Python")
    elif javascript:
        st.write(f"**Language Know:** Javascript")
    else:
        st.write(f"**Language Know:** C++")
    st.write("**Bio:** ",bio if bio else "Not Provided")"""


import streamlit as st
import pandas as pd

st.title("Dashboard")

file=st.file_uploader("Upload your file",type=['csv'])
if file:
    df=pd.read_csv(file)
    st.subheader("Top 5 rows:")
    st.write(df.head())

    st.subheader("Statistical summary:")
    st.write(df.describe())