import streamlit as st
from datetime import datetime

min_date=datetime(1990,1,1)
max_date=datetime.now()

st.title("User Information Form")
with st.form(key="user_info_form", clear_on_submit=True):
    name=st.text_input('Enter your first name')
    birth_date=st.date_input('Enter your birth date',min_value=min_date, max_value=max_date)

    if birth_date:
        age=max_date.year-birth_date.year
        if birth_date.month > max_date.month or (birth_date.month == max_date.month and birth_date.day>max_date.day):
            age-=1
        st.write(f'Your calculated age is {age}')
    
    submit_button=st.form_submit_button(label='Submit Form')

    if submit_button:
        if not name or not birth_date:
            st.warning('please fill in all form inputs')
        else:
            st.success(f"Thank you {name}, your age is {age}")
            st.balloons()