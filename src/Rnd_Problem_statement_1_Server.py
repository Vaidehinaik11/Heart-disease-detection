import streamlit as st 
import pickle
from model import SVC_MODEL_PATH
print('Successfully executed ')

import os
print(os.path.isfile(SVC_MODEL_PATH))
model = pickle.load(open(SVC_MODEL_PATH, 'rb'))

def predict(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
    prediction=model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    return prediction
    
def main():
    st.title("Problem Statement 1: Heart Disease Prediction")
    st.markdown("* Classification of a specific heart disease using machine learning techniques. ")
    st.markdown("   > Objective: To build a machine learning model, that can detect between a subject afflicted with heart disease and someone who is normal")
    
    # html_temp = """
    # <div style="background-color:tomato;padding:10px">
    # <h2 style="color:white;text-align:center;">Streamlit Heart Disease Predictor </h2>
    # </div>
    # """
    # st.markdown(html_temp,unsafe_allow_html=True)

    st.divider()
    st.subheader("Enter Data below:")
    age = st.number_input("Age",0,100, 25)
    sex = st.radio("Sex", ['Male', 'Female'])
    cp = st.selectbox("CP",['Value 0', 'Value 1', 'Value 2','Value 3'])
    trestbps = st.number_input("Tres BP",0,140)
    chol = st.number_input("Cholestrol",0,500)
    fbs=st.number_input("FBS",0,300)
    restecg = st.selectbox("Restecg",['Value 0', 'Value 1', 'Value 2'])
    thalach=st.number_input("Thalach",0,300)
    exang=st.radio("Exang", ['Yes', 'No'])
    oldpeak=st.text_input("Old Peak","Type Here")
    slope =st.selectbox("Slope",['Value 0', 'Value 1', 'Value 2'])
    ca=st.selectbox("Ca",['Value 0', 'Value 1', 'Value 2'])
    thal=st.selectbox("Thal",['Normal', 'Fixed defect', 'Reversable defect'])
    
    result=""
    if st.button("Predict"):
        try:
            columns = ['age',
                    'trestbps',
                    'chol',
                    'fbs',
                    'restecg',
                    'thalach',
                    'oldpeak',
                    'cp_0',
                    'cp_1',
                    'cp_2',
                    'cp_3',
                    'thal_0',
                    'thal_1',
                    'thal_2',
                    'slope_0',
                    'slope_1',
                    'slope_2',
                    'ca_0',
                    'ca_1',
                    'ca_2',
                    'ca_3']
            
            pd_cols_dict = {k: '' for k in columns}
            print(pd_cols_dict)
            
            # if cp:

            # result=predict(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal)
            st.success('The output is {}'.format(result))

        except Exception as e:
            st.error("error:", e)

    
        
main()
