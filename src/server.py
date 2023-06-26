import streamlit as st
import pandas as pd

# import model

from model import TestModel, TRAINED_MODEL_PATH




def predict(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal):
    prediction=model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    return prediction
    
def start_server():
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
    oldpeak=st.text_input("Old Peak","2.1")
    slope =st.selectbox("Slope",['Value 0', 'Value 1', 'Value 2'])
    ca=st.selectbox("Ca",['Value 0', 'Value 1', 'Value 2'])
    thal=st.selectbox("Thal",['Normal', 'Fixed defect', 'Reversable defect'])
    
    st.divider()

    st.header("OR")

    test_file = st.file_uploader("Upload Test data here")

    
    


    result=""
    if st.button("Predict"):
        if test_file:
            test_data_df = pd.read_csv(test_file)

        else:
            # try:
            columns = ['age',
                       'sex',
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
                    'ca_3',
                    'restecg_0',
                    'restecg_1',
                    'restecg_2']
            
            pd_cols_dict = {k: 0 for k in columns}

            if oldpeak:
                oldpeak = float(oldpeak)
            

            pd_cols_dict['age'] = age
            pd_cols_dict['trestbps'] = trestbps
            pd_cols_dict['chol'] = chol
            # pd_cols_dict['restecg'] = restecg
            pd_cols_dict['thalach'] = thalach
            pd_cols_dict['oldpeak'] = oldpeak

        
            if sex:
                if sex == "Male":
                    pd_cols_dict['sex'] = 1
                    
                elif sex == "Female":
                    pd_cols_dict['sex'] = 0
                    
            if fbs:
                if fbs > 120:
                    pd_cols_dict['fbs'] = 1
                else:
                    pd_cols_dict['fbs'] = 0
                    


            if cp:
                if cp == "Value 0":
                    pd_cols_dict['cp_0'] = 1
                if cp == "Value 1":
                    pd_cols_dict['cp_1'] = 1
                if cp == "Value 2":
                    pd_cols_dict['cp_2'] = 1
                if cp == "Value 3":
                    pd_cols_dict['cp_3'] = 1

            # if thal:
            #     if thal == "Value 0":
            #         pd_cols_dict['thal_0'] = 1
            #     if thal == "Value 1":
            #         pd_cols_dict['thal_1'] = 1
            #     if thal == "Value 2":
            #         pd_cols_dict['thal_2'] = 1
            
            if slope:
                if slope == "Value 0":
                    pd_cols_dict['slope_0'] = 1
                if slope == "Value 1":
                    pd_cols_dict['slope_1'] = 1
                if slope == "Value 2":
                    pd_cols_dict['slope_2'] = 1
            
            if ca:
                if ca == "Value 0":
                    pd_cols_dict['ca_0'] = 1
                if ca == "Value 1":
                    pd_cols_dict['ca_1'] = 1
                if ca == "Value 2":
                    pd_cols_dict['ca_2'] = 1
            
            if restecg:
                if restecg == "Value 0":
                    pd_cols_dict['restecg_0'] = 1
                if restecg == "Value 1":
                    pd_cols_dict['restecg_1'] = 1
                if restecg == "Value 2":
                    pd_cols_dict['restecg_2'] = 1
                
            if thal:
                if thal == "Normal":
                    pd_cols_dict['thal_0'] = 1
                if thal == "Fixed defect":
                    pd_cols_dict['thal_1'] = 1
                if thal == "Reversable defect":
                    pd_cols_dict['thal_2'] = 1
                
            

            print(pd_cols_dict)
            test_data_df = pd.DataFrame(pd_cols_dict, index=[0])

            st.write("Test data")
            st.dataframe(test_data_df)

        
        
        test_data_df.to_csv("../data/test/pred_df.csv", index=False)
        # print(test_data_df.values.tolist())

        test_path = "../data/test/pred_df.csv"

        
        test_model = TestModel(model_path=TRAINED_MODEL_PATH, test_data_path=test_path)

        test_model.load_model()


        predictions_df = test_model.predict()
        
        

        st.success("Predictions complete")

        st.dataframe(predictions_df)

        # except Exception as e:
        #     st.error(f"error: {e}")

    

        
start_server()

