import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder


st.sidebar.title("Enter your car's configuration here:")
html_temp = """
<div style="background-color:darkred;padding:10px">
<h2 style="color:white;text-align:center;">Estimate your car's value </h2>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)



car_model=st.sidebar.selectbox("Select model of your car", ('Audi A1', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia', 'Renault Clio', 'Renault Duster', 'Renault Espace'))
km=st.sidebar.slider("Enter km of your car", 0,350000, step=1000)
age=st.sidebar.selectbox("Enter the age of your car:",(0,1,2,3))
gearing_type=st.sidebar.radio('Select gear type',('Automatic','Manual','Semi-automatic'))
Fuel=st.sidebar.selectbox("Select fuel type:",('Benzine', 'Diesel'))


richard_model=pickle.load(open("rf_model_new","rb"))
richard_transformer = pickle.load(open('transformer', 'rb'))


my_dict = {
    "make_model":car_model,
    "km": km,
    "age": age,
    "Gearing_Type":gearing_type,
    "Fuel":Fuel    
}

df = pd.DataFrame.from_dict([my_dict])


st.subheader("The configuration that you have entered is as follows:")
st.table(df)

df2 = richard_transformer.transform(df)

st.subheader("Press ESTIMATE button below when your configuration is complete")

if st.button("ESTIMATE"):
    prediction = richard_model.predict(df2)
    st.success("The estimated value of your car is:    € {}. ".format(int(prediction[0])))
   
  
