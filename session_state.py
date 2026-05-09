import streamlit as st

if "counter" not in st.session_state:
    st.session_state.counter = 0

if st.button("Increment Counter"):
    st.session_state.counter += 1
    st.write(f"Counter incremented to {st.session_state.counter}")

if st.button("Reset"):
    st.session_state.counter = 0

st.write(f"Counter value: {st.session_state.counter}")


import streamlit as st

# Initialize only once
if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("Click me"):
    st.session_state.count += 1

st.write("Count:", st.session_state.count)

if 'step' not in st.session_state:
    st.session_state.step=1

if 'info' not in st.session_state:
    st.session_state.info={}

def go_to_step2(name):
    st.session_state.info['name']=name
    st.session_state.step=2

def go_to_step1():
    st.session_state.step=1

if st.session_state.step==1:
    st.header("Part 1:Info")

    name = st.text_input(label='Name', value=st.session_state.info.get("name",""))
    
    st.button("Next", on_click=go_to_step2, args=(name,))

if st.session_state.step==2:
    st.header("Part 2: Review")
    
    st.subheader("Please review this:")
    st.write(f'**Name**:{st.session_state.info.get('name','')}')

    if st.button("Submit"):
        st.success("Great!")
        st.balloons()
        st.session_state.info={}

    st.button("back", on_click=go_to_step1)