import streamlit as st
import pickle

st.title("Yeild Prediction APP")
Rainfall=st.number_input("Enter Rainfall",min_value=50,max_value=1000,step=1)
Temperature=st.number_input("Enter Temp",min_value=10,max_value=1000,step=1)
Fertilizer=st.number_input("Enter Ferti",min_value=50,max_value=1000,step=1)

with open('model.pkl','rb') as f:
   model= pickle.load(f)

with open('sc.pkl','rb') as f:
    sc=pickle.load(f)

if st.button("submit"):
    features=sc.transform([[Rainfall,Temperature,Fertilizer]])
    yeild=model.predict(features)[0]
    st.success(f"The predicted Yeild is {yeild:2f}")
