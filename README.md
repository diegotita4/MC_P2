# Credit Scoring Model

This project develops a **Credit Scoring Model** to evaluate the creditworthiness of clients using statistical and data science techniques.  
It includes data preprocessing, exploratory data analysis (EDA), feature engineering, and implementation of predictive models using machine learning.

---

## 🧠 Project Overview

The purpose of this project is to design and implement a **credit risk scoring system** capable of classifying clients according to their probability of default.  
The workflow follows the full **data science pipeline**, from data ingestion and cleaning to exploratory analysis and model construction.

---

## 🎯 Objectives

- Analyze and preprocess client financial data.  
- Identify key risk factors affecting credit performance.  
- Build classification models to predict credit scores using multiple machine learning algorithms.  
- Visualize results to support business interpretation.  

---

## 🏗️ Project Structure

- Analyze and preprocess client financial data.  
- Identify key risk factors affecting credit performance.  
- Build a classification model to predict credit scores.  
- Visualize results to support business interpretation.  

---

## 🏗️ Project Structure

```markdown
Modelo_Puntuacion_Crediticia/
│
├── Data/ # Raw and cleaned datasets
│ └── clean_data.xlsx
├── analysis/ # EDA scripts and notebooks
├── Models/ # Python scripts for ML models
│ ├── NN.py # Neural Network implementation
│ ├── XGBoost.py # XGBoost implementation
│ ├── benchmarkmodel.py # Logistic Regression benchmark
│ ├── decision_tree.py # Decision Tree implementation
│ ├── random_forest.py # Random Forest implementation
│ ├── data.py # Data preprocessing functions
│ └── functions.py # Helper functions
├── save_models/ # Trained model files
│ ├── DecisionTree.pkl
│ ├── LogisticRegression_benchmark.pkl
│ ├── XGBoost.pkl
│ └── credit_score_nn_model.h5
├── notebooks/ # Jupyter notebooks
│ └── Modelo_puntuacion_crediticia.ipynb
├── documentation/ # Exported reports
│ ├── Modelo_puntuacion_crediticia.docx
│ ├── Modelo_puntuacion_crediticia.html
│ └── Modelo_puntuacion_crediticia.pdf
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## ⚙️ Methodology

### Data Loading & Exploration
- Import datasets and perform initial checks on data quality, structure, and distribution.  

### Data Cleaning
- Handle missing values, outliers, and inconsistencies in the financial data.  

### Feature Engineering
- Encode categorical variables with `LabelEncoder` and create new features based on financial indicators.  

### Exploratory Data Analysis (EDA)
- Use `pandas`, `matplotlib`, and `seaborn` to visualize correlations, distributions, and patterns.  

### Modeling
- Implement multiple **classification models** to predict creditworthiness:
  - Logistic Regression (benchmark)
  - Decision Tree
  - Random Forest
  - XGBoost
  - Neural Network
- Evaluate models using appropriate metrics (accuracy, AUC, confusion matrix, etc.)  

---

## 🧩 Technologies Used

| Library               | Purpose                                    |
|-----------------------|--------------------------------------------|
| NumPy                 | Numerical operations                        |
| Pandas                | Data manipulation and analysis              |
| Matplotlib / Seaborn  | Data visualization                          |
| SciPy                 | Statistical analysis                        |
| scikit-learn          | Preprocessing and machine learning models   |
| TensorFlow / Keras    | Neural Network modeling                     |
| os                    | File and directory management               |

---

## 🚀 Installation & Usage

1. **Clone the Repository**
```bash
git clone https://github.com/diegotita4/MC_P2.git
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Results (Summary)
- Full EDA identifying main drivers of credit performance.   
- Feature transformations improving model interpretability.   
- Multiple machine learning models with saved predictions and trained models for future use.   
- Reports and visualizations available in documentation/.   
> Detailed metrics and model evaluation can be found in the notebook.

---

## 👥 Authors
**Project:** _Credit Scoring Model_   
**Team:**
- Mugica Liparoli Juan Antonio
- Enríquez Nares Diego Emilio
- Brizuela Casarín Ana Sofía   
**Course:** _Credit Models_   
**Professor:** _Rodolfo Slay Ramos_   
**Date:** _September 24, 2024_   

---

## 🪪 License
This project is for academic and educational purposes only.
_All rights reserved © 2024 — Credit Scoring Model Team._
