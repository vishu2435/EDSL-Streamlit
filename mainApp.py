import train_models
import predict
import streamlit as st

PAGES ={
    "Train Model":train_models,
    "CHeck Predictions ": predict
}

st.sidebar.title("Navigation")

selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()