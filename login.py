import streamlit as st
from datetime import datetime

st.title("User information")

form_values={'Name':None,
             'Height':None,
             'Gender':None,
             'Dob':None}

min_date=datetime(1990,1,1)
max_date=datetime.now()

with st.form(key='user_info_form'):
    form_values['Name']=st.text_input('Enter your name: ')
    form_values['Height']=st.number_input('Enter your height: ')
    form_values['Gender']=st.selectbox('Gender',['Male','Female','Other'])
    form_values['Dob']=st.date_input('Enter your birthdate',min_value=min_date,max_value=max_date)

    submit_button=st.form_submit_button(label='Submit')

    if submit_button:
        if not all(form_values.values()):
            st.warning("Please fill in all of the fields")
        else:
            st.balloons()
            st.write('### Info')
            for (key, value) in form_values.items():
                st.write(key,":",value)

