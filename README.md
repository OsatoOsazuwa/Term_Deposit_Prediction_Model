# Term_Deposit_Prediction_Model
This project aims to predict whether a customer will subscribe to a term deposit based on various features such as demographics, job type, contact communication type, and previous campaign results. It utilizes machine learning techniques, specifically a **Neural Network** model, to perform the classification task.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Libraries and Dependencies](#libraries-and-dependencies)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Building](#model-building)
6. [Model Evaluation](#model-evaluation)
7. [Results](#results)
8. [How to Run](#how-to-run)
9. [Saved Model](#saved-model)

## Project Overview
The goal of this project is to build a machine learning model to predict whether a customer will subscribe to a term deposit (`"yes"` or `"no"`) based on demographic and behavioral features. The model is built using a **Neural Network** and evaluated using classification metrics like **accuracy**, **precision**, **recall**, and **F1-score**.

## Dataset
The dataset is sourced from the **UCI Machine Learning Repository**, and it contains information about customer attributes, their responses to previous marketing campaigns, and whether or not they subscribed to a term deposit.

- **Target Variable**: `deposit` (Binary: 0 = "no", 1 = "yes")
- **Features**: Various demographic and behavioral features such as age, job, marital status, education, default status, housing loan, contact communication type, month of last contact, and previous campaign outcome.

## Libraries and Dependencies
The following libraries are required for this project:
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical computations.
- **matplotlib** and **seaborn**: For data visualization.
- **scikit-learn**: For machine learning functions such as model evaluation and preprocessing.
- **tensorflow**: For building the Neural Network model.

You can install the dependencies by running the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Data Preprocessing
1. **Handling Categorical Variables**: We used **One-Hot Encoding** to convert categorical variables into numeric values for the model.
2. **Feature Selection**: The target variable `deposit` was separated from the feature set. We used `StandardScaler` to normalize the features for the neural network.
3. **Train-Test Split**: The dataset was split into training (85%) and testing (15%) sets to evaluate model performance.
   
## Model Building
We built a **Neural Network** using the `Sequential` API from TensorFlow. The architecture consists of:
- Input layer: 64 neurons with ReLU activation
- Hidden layer: 32 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation for binary classification

We used the **Adam** optimizer and **binary cross-entropy loss** function for model compilation.

### Code for Building the Model:
```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## Model Evaluation
After training the model, we evaluated its performance on the test set using various metrics:
- **Accuracy**: 84%
- **Precision, Recall, F1-score**: Calculated for both classes (0 = "no", 1 = "yes") using the `classification_report` from **scikit-learn**.

### Model Evaluation Results:
```plaintext
precision    recall  f1-score   support

           0       0.89      0.80      0.84       879
           1       0.80      0.89      0.84       796

    accuracy                           0.84      1675
   macro avg       0.84      0.84      0.84      1675
weighted avg       0.85      0.84      0.84      1675
```

## Results
The model achieved an overall **accuracy of 84%**. Precision and recall for class 1 (deposit) were balanced, with a recall of 89%, indicating that the model successfully identifies deposit subscribers. The F1-scores for both classes are around **0.84**, showing good overall performance.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/OsatoOsazuwa/bank-deposit-prediction.git

## Saved Model
The trained model has been saved as a .keras file. You can load it and use it to make predictions on new data.

Example of loading the saved model:
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('bank_deposit_model.h5')

# Make predictions with new data
predictions = model.predict(X_new_data)


