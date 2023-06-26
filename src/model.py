"""Created by Vishak G.

Datetime: 17-05-2023

Description: Helper file to train model, and perform predictions 
"""

# Import internal libraries
import os
import datetime
import pickle

# Import External libraries
import pandas as pd
import numpy as np

# for model building
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score

SCALER_SAVE_PATH = "../models/saved/scaler_model.sav"
TRAINED_MODEL_PATH = "../models/saved/trained_model.pkl"


def scale_data(df, target_col=None, test=None):
    """Scale data using Sklearn MinMaxScaler

    Args:
        test_data (array/list): List of input values for test

    Returns:
        array/list: Returns test data after performing MinMaxScaling operation
    """

    # Open existing model file, if any
    if os.path.isfile(SCALER_SAVE_PATH):
        with open(SCALER_SAVE_PATH, 'rb') as scaler_model:
            scaler = pickle.load(scaler_model)
    else:
        scaler = StandardScaler()
        scaler.fit(df)

    # If target col passed as value, drop the column for standardization
    if target_col:
        scaled_test_data = scaler.transform(df.drop(target_col))
    else:
        scaled_test_data = scaler.transform(df)


    if not test:
        print("inside not test")
        # Store the scaler model instance
        with open(SCALER_SAVE_PATH, 'wb') as f:
            pickle.dump(scaler, f)


    return scaled_test_data

class TrainModel:

    def __init__(self,train_data_path, test_data_path, target_col, model, model_path):
        """Initialize Class with parameter values

        Args:
            train_data_path (str): Path for Training data
            test_data_path (str): Path for Test Data
            target_col (str): Name of the column to predict from the training data
            model (any): Instance of the Ml Model to be used
            model_path (str): Save path for the trained model
        """

        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.model = model
        self.target_col = target_col
        self.model_path = model_path

        return

    def load_train_data(self):
        """Load Training Data

        Returns:
            pandas.DataFrame: return Train data (Dataframe)
        """
        return pd.read_csv(self.train_data_path)

    def encode_target_column(self, y):
        """Function to encode target column

        Args:
            y (pd.Series): List of target column values

        Returns:
            pd.Series: Return the target encoded values (Eg: 0 -> No Disease, 1->Disease)
        """
        le = LabelEncoder().fit(y)
        y = le.transform(y)

        self.le = le

        return y
    
    def perform_preprocessing(self,df):
        """Function to perform preprocessing of data

        Args:
            df (pd.DataFrame): DataFrame to preprocess

        Returns:
            df (pd.DataFrame): returns dataframe after cleaning, feature engineering
        """

        # Change datatype of column based on reference pdf
        all_catg_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

        # Convert str column to object datatype
        df[all_catg_cols] = df[all_catg_cols].astype(object)

        # print(df.head())

        # Convert categorical columns to Numeric (using one-hot encoding) 
        catg_cols = ['thal', 'ca', 'exang', 'cp', 'slope']

        cp_df = pd.get_dummies(df['cp'],dtype=int,prefix = 'cp')
        thal_df = pd.get_dummies(df['thal'],dtype=int, prefix='thal')
        slope_df = pd.get_dummies(df['slope'], dtype=int, prefix='slope')
        ca_df = pd.get_dummies(df['ca'], dtype=int,prefix='ca')
        rest_ecg_df = pd.get_dummies(df['restecg'], dtype=int,prefix='restecg')
        

        df = pd.concat([df, cp_df, thal_df, slope_df, ca_df,rest_ecg_df], axis=1)
        df = df.drop(columns=catg_cols)
        
        return df

    def split_train_data(self, X, y):
        """Function to split train, test data

        Args:
            X (pd.DataFrame): All values except target column to be used for prediction
            y (pd.DataFrame): Target column values

        Returns:
            X_train, X_test, y_train, y_test (pd.DataFrame): Dataframe after splitting into train, test data 
        """
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1,stratify=y)

        self.model.feature_names = X.columns

        print(X_test)
        pd.concat([X_test, pd.Series(y_test)],axis=1).to_csv("tmp.csv")

        return X_train, X_test, y_train, y_test


    def train_model(self, X_train, X_test, y_train, y_test):
        """Function to train model based on Train, test data

        Args:
            X_train (pd.DataFrame): Train dataset
            X_test (pd.DataFrame): Test dataset
            y_train (pd.DataFrame): Train target column
            y_test (pd.DataFrame): Test target column
        """

        self.model.fit(X_train, y_train)

        # Predict the response for test dataset
        y_train_pred = self.model.predict(X_train)
        y_pred = self.model.predict(X_test)
        model_train_acc = accuracy_score(y_train, y_train_pred)
        model_test_acc =accuracy_score(y_test, y_pred)

        
    

        print("Model Training Accuracy", model_train_acc )
        print("Model Test Accuracy", model_test_acc)

        print("Confusion Matrix : \n",confusion_matrix(y_test,y_pred),'\n\n')
        print(classification_report(y_test,y_pred))

        self.save_trained_model()

        return

    def save_trained_model(self):
        """Save trained model as pkl file

        Returns:
            bool: True once process is completed
        """

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)

        return True

    def start_process(self):
        """Function to start full pipeline 
         1. Data loading
         2. Data preprocessing
         3. Data cleaning 
         4. Feature engineering
         5. Train/test split
         6. Model buildiing
         7. Model serialization
        """

        print("Started")
        
        print("Loading Data")
        
        df = self.load_train_data()
        
        print("Data loaded successfully")

        print("Data preprocessing started")
        preprocessed_df = self.perform_preprocessing(df)
        print("Data preprocessing complete")

        X, y = preprocessed_df.drop(self.target_col, axis=1), preprocessed_df[self.target_col]
        
        # print("Encoding target column")
        # y = self.encode_target_column(y)

        # self.model.target_names = self.return_label_map() 
        self.model.target_names = {0:0, 1:1} 

        print("Target cols", self.model.target_names)

        print("Splitting data to train, test..")
        X_train, X_test, y_train, y_test = self.split_train_data(X, y)

        print(X.head())
        print("Scaling data")
        X_train = scale_data(X_train)
        X_test = scale_data(X_test)

        print("Training started")
        self.train_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        self.save_trained_model()

        print("Model saved")

        return
        
    def return_saved_model_path(self):
        return self.model_path
    
    def return_label_map(self):
        # Return Class mapping
        return dict(zip(self.le.classes_, self.le.transform(self.le.classes_)))



class TestModel:
    def __init__(self, model_path, test_data_path, scaler_model_path=None):
        self.model_path = model_path
        self.test_data_path = test_data_path
        
        if model_path:
            self.model_path = model_path

        return
    
    def load_model(self):
        """Load serialized ML model (.pkl)
        """

        with open(self.model_path, 'rb') as model_file:
            model = pickle.load(model_file)

        print(model.feature_names)
        print(model.target_names)

        return model


    def predict(self, scaler_model=None):
        """Predict on test data - based on saved model

        Args:
            test_data (list/array): Array

        Returns:
            str: Returns predicted label for given test data
        """

        pred_df = pd.read_csv(self.test_data_path)

        
        model = self.load_model()
        labels = model.target_names

        scaled_data = scale_data(pred_df,test=True)
    
        pred_df['Predictions'] = model.predict(scaled_data)
    
        
        # if scaler_model:
        # else:
        #     pred_df['Predictions'] = model.predict(pred_df)
            
        # print(pred_df)

        pred_df['Predictions_label'] = pred_df['Predictions'].apply(lambda x: labels[x])
        
        print(pred_df)

        # print("Feature names",svc_model.feature_names)
        return pred_df
    

if __name__ == "__main__":


    train_path = "../data/train/heart_cleveland_upload.csv"
    test_path = "../data/test/pred_class_zero.csv"

    # print(os.path.isfile(train_path))

    svm = SVC()

    # temp = f"../models/saved/{'hello'}.pkl"
    
    train_model = TrainModel(
        train_data_path=train_path,
        test_data_path=test_path,
        model=svm,
        target_col='condition',
        model_path=TRAINED_MODEL_PATH
        )
    
    train_model.start_process()

    