import streamlit as st
st.sidebar.title("This is the sidebar")
st.sidebar.write("You can place elements like sliders, buttons, and text here")
siderbar_input=st.sidebar.text_input("Enter something in the sidebar")

tab1, tab2, tab3 = st.tabs(["Tan 1", "Tab 2", "Tab 3"])

with tab1:
    st.write("You are in Tab1")

with tab2:
    st.write("You are in tab 2")

with tab3:
    st.write("You are in tab 3")

col1, col2, col3 = st.columns(3)
with col1:
    st.header("Column 1")
    st.write("Content for column 1")

with col2:
    st.header("Column 2")
    st.write("Content for column 2")

with col3:
    st.header("Column 3")
    st.write("Content for column 3")

with st.container(border=True):
    st.write("This is inside a container")
    st.write('you can think of containers as a grouping for elements')

st.button('tooltip',help='its a tooltip')