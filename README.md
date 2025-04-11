# 🛸 Machine Failure Prediction App

This project is a Streamlit-based web application that leverages machine learning to predict machine failure using sensor data. The app is designed to provide an interactive way for users to upload datasets, explore their data, train a machine learning model, and make real-time predictions—all within a clean and user-friendly interface.

## 🔍 Project Insights

- **Interactive Dashboard**: Users can upload CSV files, explore raw data, view correlation heatmaps, inspect features and target variables, and perform complete ML workflows.
- **Accuracy Achieved**: The model currently achieves an accuracy of **64%**, which is acceptable for initial experimentation but indicates potential for improvement.
- **Possible Limitation**: The relatively low accuracy is likely due to the **small size of the dataset**, which can hinder the model’s ability to generalize well.
- **Data Processing**: Automated label encoding and feature standardization prepare the data for modeling.
- **Prediction Interface**: Users can manually input new data to receive a live prediction for machine failure.

## ⚙️ How It Works

1. **Data Upload**: Upload your machine failure dataset in CSV format.
2. **Data Exploration**: Understand your data with tools for summary statistics, data types, and visual correlation matrices.
3. **Preprocessing**: Label encodes categorical data and scales numeric features.
4. **Model Training**: Trains a Random Forest Classifier using a train/test split.
5. **Evaluation**: Generates classification report, confusion matrix, and accuracy score.
6. **Real-Time Prediction**: Accepts user inputs and returns a prediction with balloon celebration for "No Machine Failure".

## 🧗 Challenges Faced

1. **Limited Dataset Size**:
   - **Issue**: Small dataset size led to relatively low accuracy (~64%).
   - **Solution**: Despite the limitation, we focused on thorough preprocessing and model evaluation. The app remains useful for educational or exploratory analysis and is ready for improved data.

2. **Dynamic Input Handling**:
   - **Issue**: Ensuring the prediction system correctly transformed user input to match training data format.
   - **Solution**: Reused trained `LabelEncoder` and `StandardScaler` to transform real-time input data before predictions.

3. **Visual Interpretability**:
   - **Issue**: Non-technical users may struggle with raw metrics.
   - **Solution**: Added visualizations like confusion matrices and correlation heatmaps to simplify interpretation.

## 🚀 Future Enhancements

- 🧠 **Advanced Modeling**: Support for other classifiers (e.g., SVM, XGBoost, Neural Nets).
- 🔁 **Larger Datasets**: Improve performance by integrating with larger or live datasets.
- 🧪 **Cross-Validation**: Add k-fold cross-validation for better evaluation.
- 🎯 **Feature Selection**: Automatically identify and use the most important features.
- 💾 **Model Persistence**: Save and reload trained models for reuse.
- 🖼️ **UI Improvements**: Add light/dark mode toggle and improved layout for mobile responsiveness.
