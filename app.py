import streamlit as st
import pandas as pd
import pickle as pkl


st.title("Diabetic Patient Classification")
st.write("Welcome to Pruthviraj chavan's Project")
pipe=pkl.load(open("pipe.pkl","rb+"))
model=pkl.load(open("model.pkl","rb+"))
scaler=pkl.load(open("scaler.pkl","rb+"))
ds=pd.read_csv("cleaned.csv")

#'Pregnancies' 
# 'Glucose'
# 'BloodPrDessure'
# 'SkinThickness'
# 'Insulin'
# 'BMI'
# 'DiabetesPedigreeFunction'
# 'Age'
pregnancies=st.number_input("Enter the no of pregnancies(0-20)",min_value=0,max_value=20,step=1)
glucose=st.number_input("Enter the Glucose Level:",min_value=70,step=1)
bloodpressure=st.number_input("Enter the BloodPressure Level:",min_value=60,max_value=200,step=1)
Skinthickness=st.number_input("Enter the SkinThickness: Level:",min_value=10,max_value=50,step=1)
insulin=st.number_input("Enter the Insulin Level:",min_value=16,max_value=166,step=1)
bmi=st.number_input("Enter the BMI of Your Body:",min_value=18.5,max_value=40.0,step=0.01)
diabetespedigreeFunction=st.number_input("Enter the Diabetes Pedigree Function Level:",min_value=0.0,max_value=2.5,step=0.01)
bloodpressure=st.number_input("Enter the Age::",min_value=20,max_value=100,step=1)

if st.button('Predict'):
    columns=['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']
    data=[[5,105,80,40,150,40,0.99,50]]
    myinput=pd.DataFrame(data=data,columns=columns)
    result=pipe.predict(myinput)
    if result[0]==1:
        st.write("Yes Patient is Diabetic")
    else:
        st.write("Patient is not Diabetic")
    
    
