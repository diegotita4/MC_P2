# --------------------------------------------------------------------------------------------------- #
# -- project: Credit Score Prediction decision_tree                                  -- #
# -- script: decision_tree.py : python script with the decision_tree model functionality               -- #
# -- author: YOUR GITHUB USER NAME                                                                   -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                           -- #
# -- repository: YOUR REPOSITORY URL                                                                 -- #
# --------------------------------------------------------------------------------------------------- #

# Importing Libraries
import os
import pickle
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Ensure Models folder exists
os.makedirs('Models', exist_ok=True)

# Load and Split Data
def load_data(file_path):
    print("Loading data...")
    try:
        data = pd.read_excel(file_path)
        print(f"Data loaded successfully from {file_path}.")
        logging.info(f"Data loaded successfully from {file_path}")
        
        # Limit the data to 90%
        limited_data = data.sample(frac=1, random_state=42)
        print(f"Using 90% of the dataset, {len(limited_data)} rows.")
        
        print("Splitting data into training and testing sets...")
        train_data, test_data = train_test_split(limited_data, test_size=0.30, random_state=42)
        print("Data split completed.")
        logging.info("Data split into train and test sets")
        return train_data, test_data
    except Exception as e:
        print(f"Error loading data: {e}")
        logging.error(f"Error loading data: {e}")
        raise

# Preprocessing
def preprocess_data(train_data, test_data):
    print("Preprocessing data...")
    
    X_train = train_data.drop(columns=["Credit_Score"])
    y_train = train_data["Credit_Score"]
    X_test = test_data.drop(columns=["Credit_Score"])
    y_test = test_data["Credit_Score"]

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Data preprocessing completed.")
    logging.info("Data preprocessing completed")
    return X_train, X_test, y_train, y_test

# Train Decision Tree Model
def train_decision_tree(X_train, y_train):
    print("Training Decision Tree model...")
    
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)

    print("Decision Tree model training completed.")
    logging.info("Decision Tree model trained successfully")

    return dt_clf

# Model Evaluation and Saving (with plotting)
def evaluate_and_save_model(model, X_test, y_test):
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Model Accuracy: {accuracy}")
    print(f"Model F1 Score: {f1}")
    
    logging.info(f"Model Accuracy: {accuracy}")
    logging.info(f"Model F1 Score: {f1}")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1, 2])
    disp.plot()
    plt.title("Confusion Matrix for Decision Tree Model")
    plt.show()

    # AUC-ROC Curve for each class
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        auc_score = roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc_score:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("AUC-ROC Curve for Each Class")
    plt.legend()
    plt.show()
    
    # Error over time (training iterations)
    errors = []
    for i in range(1, len(y_test) + 1):
        y_pred_partial = model.predict(X_test[:i])
        error_rate = np.mean(y_pred_partial != y_test[:i])
        errors.append(error_rate)
    
    plt.plot(errors)
    plt.xlabel("Number of Samples")
    plt.ylabel("Error Rate")
    plt.title("Error Rate Over Time (Samples)")
    plt.show()

    # Save the model using pickle
    model_path = 'Models/DecisionTree_benchmark.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved successfully to {model_path}.")
    logging.info(f"Model saved to {model_path}")

    return accuracy, f1

# Main pipeline function
def run_pipeline(file_path):
    print("Starting the pipeline...")

    # Load and preprocess data
    train_data, test_data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(train_data, test_data)
    
    # Train Decision Tree model
    model = train_decision_tree(X_train, y_train)
    
    # Evaluate and save the model
    accuracy, f1 = evaluate_and_save_model(model, X_test, y_test)
    
    print("Pipeline completed.")
    return accuracy, f1

# Path to the dataset (to be passed as argument)
data_path = "Data/clean_data.xlsx"

# Example run (this is where the benchmark model is trained and evaluated)
accuracy, f1 = run_pipeline(data_path)
print(f"Accuracy: {accuracy}, F1 Score: {f1}")
