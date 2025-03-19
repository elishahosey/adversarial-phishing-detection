import streamlit as st
subject = st.text_input("Subject line")
email = st.text_area("Email")

predict= st.button("PREDICT")