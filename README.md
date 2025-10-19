# Credit Scoring Model

This project develops a **Credit Scoring Model** that evaluates the creditworthiness of clients using statistical and data science techniques.  
The model applies data preprocessing, exploratory data analysis (EDA), and feature engineering to generate a predictive scoring system that supports financial decision-making.

---

## 🧠 Project Overview

The purpose of this project is to design and implement a **credit risk scoring system** capable of classifying clients according to their probability of default.  
It follows a complete data science workflow, from data ingestion and cleaning to exploratory analysis and model construction.

---

## 🎯 Objectives

- Analyze and preprocess client financial data.  
- Identify key risk factors affecting credit performance.  
- Build a classification model to predict credit scores.  
- Visualize results to support business interpretation.  

---

## 🏗️ Project Structure

Modelo_Puntuacion_Crediticia/
│
├── data/ # Raw and processed datasets
├── notebooks/ # Exploratory and modeling Jupyter notebooks
├── src/ # Scripts for data processing and model building
├── results/ # Reports, visualizations, and model outputs
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## ⚙️ Methodology

1. **Data Loading & Exploration**  
   Import datasets and perform initial checks on data quality, structure, and distribution.

2. **Data Cleaning**  
   Handle missing values, outliers, and data inconsistencies.

3. **Feature Encoding**  
   Apply `LabelEncoder` to categorical variables for model readiness.

4. **Exploratory Data Analysis (EDA)**  
   Use `pandas`, `matplotlib`, and `seaborn` for visual correlation and distribution analysis.

5. **Feature Engineering**  
   Create new relevant features based on financial indicators.

6. **Modeling (optional future step)**  
   Implement machine learning algorithms (e.g., Logistic Regression, Random Forest) to generate the credit score.

---

## 🧩 Technologies Used

| Library | Purpose |
|---------|---------|
| **NumPy** | Numerical operations |
| **Pandas** | Data manipulation and analysis |
| **Matplotlib / Seaborn** | Data visualization |
| **SciPy** | Statistical analysis |
| **scikit-learn** | Preprocessing and modeling |
| **os** | File and directory management |

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Modelo_Puntuacion_Crediticia.git
cd Modelo_Puntuacion_Crediticia
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Analysis
Open the main notebook or run the scripts inside the src/ folder:

```bash
jupyter notebook notebooks/credit_scoring.ipynb
```

## 📊 Results (Summary)
Comprehensive EDA identifying main drivers of credit performance.

Feature transformations improving model interpretability.

Scoring framework suitable for future predictive modeling.

(Detailed metrics and model evaluation can be found in the notebooks.)

👥 Authors
**Project:** _Credit Scoring Model_
**Team:**
- Mugica Liparoli Juan Antonio
- Enríquez Nares Diego Emilio
- Brizuela Casarín Ana Sofía
**Course:** _Credit Models_
**Professor:** _Rodolfo Slay Ramos_
**Date:** _September 24, 2024_

## 🪪 License
This project is for academic and educational purposes only.
_All rights reserved © 2024 — Credit Scoring Model Team._
