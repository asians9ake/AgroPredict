
# AgroPredict

## ➲ Project description
This project develops a crop recommendation system based on soil characteristics and environmental conditions. The model is trained on a dataset containing various features such as nitrogen, phosphorus, potassium levels, temperature, humidity, pH, and rainfall. Multiple machine learning algorithms, including Random Forest, Decision Tree, SVM, Logistic Regression, and several deep learning models like LSTM, ANN, RNN, and CNN, are used to predict the most suitable crop for a given set of conditions. The final prediction is made using an ensemble model based on majority voting.

<img src="https://img.shields.io/badge/Language:-Python-5555ff">  <img src="https://img.shields.io/badge/Platform:- Google Colab-E32800">

## ➲ DATASET
The dataset contains features relevant to crop growth and yield. The data includes:
- N (Nitrogen)
- P (Phosphorus)
- K (Potassium)
- Temperature
- Humidity
- pH
- Rainfall
- Crop Label (target)

## ➲ STEPS
1. Data Analysis and Visualization
2. Data Preprocessing
3. Model Training and Evaluation:
    - Random Forest
    - Decision Tree
    - SVM
    - Logistic Regression
    - LSTM
    - ANN
    - RNN
    - CNN
4. Hyperparameter Tuning (Random Forest)
5. Model Ensemble and Prediction
6. Predicting Crop for New Data Points

## ➲ RESULTS
The accuracy scores and classification reports for each model are provided, with the final ensemble model yielding the best performance. Below are the results for each individual model:

- **Random Forest Accuracy:** 
- **Decision Tree Accuracy:** 
- **SVM Accuracy:** 
- **Logistic Regression Accuracy:** 
- **LSTM Accuracy:** 
- **ANN Accuracy:** 
- **RNN Accuracy:** 
- **CNN Accuracy:** 
- **Ensemble Model Accuracy:** 

## ➲ USAGE
To use this project, follow these steps:
1. Clone the repository.
2. Install the required dependencies.
3. Run the provided code.

## ➲ EXAMPLE
Here is an example of how to use the trained models to predict the best crop for a given set of conditions:

```python
# Column names from the original dataset
column_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# New data point
data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])

# Convert the new data point to a DataFrame with appropriate column names
data_df = pd.DataFrame(data, columns=column_names)

# Scale the new data point for the LSTM model
data_scaled = scaler.transform(data_df)
data_scaled = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)

# Get predictions from each model on the new data point
rf_prediction = rf_model.predict(data_df)
dt_prediction = dt_model.predict(data_df)
lr_prediction = lr_model.predict(data_df)
svm_prediction = svm_model.predict(data_df)
lstm_prediction = lstm_model.predict(data_scaled).argmax(axis=1)  # LSTM output needs to be transformed to class
ann_prediction = ann_model.predict(data_df)
rnn_prediction = rnn_model.predict(data_scaled).argmax(axis=1)
cnn_prediction = cnn_model.predict(data_scaled).argmax(axis=1)

# Stack the predictions
predictions = np.column_stack((rf_prediction, dt_prediction, lr_prediction, svm_prediction, lstm_prediction, ann_prediction, rnn_prediction, cnn_prediction))

# Perform majority voting
final_prediction, _ = mode(predictions, axis=1)

# Map the numerical label back to the original crop name
final_crop_name = label_encoder.inverse_transform(final_prediction.ravel())

print("Predicted Crop:", final_crop_name[0])
```
