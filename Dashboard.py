import streamlit as st
import os
st.title("Reshma Cloth Center")
st.image(os.path.join(os.getcwd(),"static","Cloths.jpg"))

tab1,tab2,tab3,tab4,tab5,tab6= st.tabs(['Dashboard', 'Inventory', 'Sales', 'Purchase', 'Purchase_List', 'Suppliers'])
with tab1:
    st.sidebar.title("Reshma Cloth Center")
    st.sidebar.button("Dashboard")
    st.sidebar.button("Inventory")
    st.sidebar.button("Sales")
    st.sidebar.button("Purchase")
    st.sidebar.button("Purchase List")
    st.sidebar.button("Suppliers")

    st.header("Dashboard")
    st.subheader("Inventory")
    col1, col2,col3=st.columns(3)
    with col1:
        st.write("In Stock")
    with col2:
        st.write("Out Of Stock")
    with col3:
        st.write("Pending Items")

    st.subheader("Sales")
    col1, col2,col3=st.columns(3)
    with col1:
        st.write("Highest Sales")
    with col2:
        st.write("Profit Per Group")
    with col3:
        st.write("Sales List")

    st.subheader("Purchase")
    col1, col2,col3=st.columns(3)
    with col1:
        st.write("Products recently purchased")

    st.subheader("Purchase List")
    col1, col2,col3=st.columns(3)
    with col1:
        st.write("Products to order")

    st.subheader("Suppliers")
    col1, col2,col3=st.columns(3)
    with col1:
        st.write("List of suppliers")

    st.subheader("Group of Products")
    col1, col2,col3, col4, col5, col6=st.columns(6)
    with col1:
        st.write("Dresses")
    with col2:
        st.write("Saree")
    with col3:
        st.write("Suits")
    with col4:
        st.write("Shirts")
    with col5:
        st.write("Pants")
    with col6:
        st.write("Inner wears")

with tab2:
    st.header("Inventory")
    st.subheader("Stock")
    col1, col2,col3, col4, col5, col6=st.columns(6)
    with col1:
        st.button("Dresses", key='col1btn2')
    with col2:
        st.button("Saree", key='col2btn2')
    with col3:
        st.button("Suits", key='col3btn2')
    with col4:
        st.button("Shirts", key='col4btn2')
    with col5:
        st.button("Pants", key='col5btn2')
    with col6:
        st.button("Inner wears", key='col6btn2')

    st.subheader("Out of Stock")
    col1, col2,col3, col4, col5, col6=st.columns(6)
    with col1:
        st.button("Dresses", key='col1btn3')
    with col2:
        st.button("Saree", key='col2btn3')
    with col3:
        st.button("Suits", key='col3btn3')
    with col4:
        st.button("Shirts", key='col4btn3')
    with col5:
        st.button("Pants", key='col5btn3')
    with col6:
        st.button("Inner wears", key='col6btn3')

    st.subheader("Pending Items")
    col1, col2,col3, col4, col5, col6=st.columns(6)
    with col1:
        st.button("Dresses")
    with col2:
        st.button("Saree")
    with col3:
        st.button("Suits")
    with col4:
        st.button("Shirts")
    with col5:
        st.button("Pants")
    with col6:
        st.button("Inner wears")

