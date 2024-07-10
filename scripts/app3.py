import streamlit as st
import time


record = st.toggle("Stop Recording/Start Recording")


if record:
    while record:
        print(time.sleep(2))
        st.write("Recording...")
        if not record:
            st.write("lets goooo")
