import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# creating the title
st.title("Machine Failure App🛸")
st.subheader('Predicting Machine Failure with Streamlit')
st.write(
    """This app predicts machine failure using a dataset of machine sensor data.
    The dataset contains various features that can be used to predict the likelihood of machine failure.""")

# Load the dataset
if st.checkbox('Show raw data'):
    df = pd.read_csv('machine_failure_dataset.csv')
    st.write(df.head())

# Make an uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

with st.expander('Data Loading and Understanding'):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    
    # Show the first 5 rows of the dataset
        if st.checkbox('Show Dataframe'):
            st.write(df.head())

        #Show the data types of the columns
        if st.checkbox('Show Data Types'):
            st.write(df.dtypes)

        # Show the summary statistics of the columns
        if st.checkbox('Show Summary Statistics'):
            st.write(df.describe())
            st.write(df.describe(include='object'))
        
        # Show the correlation matrix
        if st.checkbox('Show Correlation Matrix'):
            corr = df.select_dtypes(exclude='object').corr()
            plt.figure(figsize=(10,8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Correlation Matrix')
            st.pyplot(plt)

# Splitting the data into features and target variable
with st.expander('Features vs Target Variables'):
    if uploaded_file is not None:
        # Split the data into features and target variable
        X = df.iloc[:, :-1]  # All columns except the last one
        y = df.iloc[:, -1] # Last column

        # Show the features and target variable
        if st.checkbox('Show Features'):
            st.write(X.head())
        if st.checkbox('Show Target Variable'):
            st.write(y.head())
        
        # Show the distribution of the target variable
        if st.checkbox('Show Target Variable Distribution'):
            plt.figure(figsize=(10,6))
            sns.countplot(x=y)
            plt.title('Target Variable Distribution')
            st.pyplot(plt)

# Label Encoder
with st.expander('Data Preprocessing'):
    if uploaded_file is not None:
        # Encode the X using Label Encoder
        le = LabelEncoder()
        for column in X.select_dtypes(include=['object']).columns:
            X[column] = le.fit_transform(X[column])

        # Standardizing the collumns
        scale = StandardScaler()
        X = scale.fit_transform(X)
        st.write(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Show the shape of the training and testing sets
        if st.checkbox('Show Training Set Shape'):
            st.write('X_train shape:', X_train.shape)
            st.write('y_train shape:', y_train.shape)
        if st.checkbox('Show Testing Set Shape'):
            st.write('X_test shape:', X_test.shape)
            st.write('y_test shape:', y_test.shape)

# Model Training
if uploaded_file is not None:
    with st.expander('Model Training'):
        # Train the Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = rf.predict(X_test)

        # Show the classification report
        if st.checkbox('Show Classification Report'):
            st.text(classification_report(y_test, y_pred))

        # Show the confusion matrix
        if st.checkbox('Show Confusion Matrix'):
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10,6))
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            st.pyplot(plt)

        # Show the accuracy score
        if st.checkbox('Show Accuracy Score'):
            accuracy = accuracy_score(y_test, y_pred)
            st.write('Accuracy:', accuracy)

# Building Predictive system (Input Features)
with st.sidebar:
    st.write('Enter the input features to predict machine failure:')
    # Create input fields for the features
    input_features = {}
    for column in df.columns[:-1]:
        if df[column].dtype == 'object':
            input_features[column] = st.selectbox(column, df[column].unique())
        else:
            # For other data types, use a number input field    
            input_features[column] = st.number_input(column, value=0.0)

    # Show the input features
    if st.button('Predict'):
        input_df = pd.DataFrame([input_features])
        # Encode the categorical features
        for column in input_df.select_dtypes(include=['object']).columns:
            input_df[column] = le.transform(input_df[column])
        # Encode the input features using Label Encoder
        input_df = scale.transform(input_df)
        prediction = rf.predict(input_df)
        if prediction[0] == 1:
            st.write('Machine Failure')
        else:
            st.write('No Machine Failure')
            st.balloons()
        
