# Road Accident Severity Prediction

A data science project analyzing the **US Accidents (Kaggle)** dataset to plan, design, and document a full pipeline for predicting road accident severity using Python. This repository was developed as part of the **Virtual Data Science Apprentice – Python Specialist Intern** program (4-week internship, 10-Aug-2026 to 07-Sep-2026).

> **Note:** This project focuses on end-to-end planning and documentation of a data science workflow — dataset scoping, data cleaning strategy, exploratory analysis strategy, and model selection/evaluation strategy — produced as structured Word (.docx) reports for each weekly milestone.

---

## 📊 Dataset

**Source:** [US Accidents (Kaggle)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)

A large-scale, countrywide traffic accident dataset covering multiple US states, including accident severity, weather conditions, road features, time of day, and geospatial information. The severity target is treated as a 3-class classification problem: **Minor, Serious, Fatal**.

---

## 🗂️ Project Structure
road-accident-severity-prediction/
├── week1/
│ └── Week1_Project_Planning_and_Dataset_Scoping.docx
├── week2/
│ └── Week2_Data_Cleaning_and_Transformation_Documentation.docx
├── week3/
│ └── Week3_EDA_and_Visualization_Strategy.docx
├── week4/
│ └── Week4_ML_Model_Selection_and_Evaluation_Plan.docx
└── README.md


---

## 📅 Weekly Deliverables

### Week 1: Project Planning and Dataset Scoping


This report defines the project objectives, scope, and anticipated challenges for the Road Accident Severity Prediction project. It identifies the US Accidents (Kaggle) dataset as the primary data source and explains the rationale for its selection, including its scale, feature richness, and relevance to real-world road safety problems. The document outlines a high-level analytical workflow — from data acquisition through cleaning, exploratory analysis, modeling, and reporting — along with the Python tools and libraries planned for each stage.

### Week 2: Data Cleaning and Transformation Documentation


This report documents a detailed data cleaning and transformation strategy for the US Accidents dataset. It covers techniques for handling missing values, duplicates, and outliers using pandas and numpy, along with a plan for feature scaling, normalization, and feature engineering. The document includes worked examples of cleaning procedures and lays the groundwork for a data pipeline that produces analysis-ready data for the subsequent exploratory and modeling phases.

### Week 3: Exploratory Data Analysis and Visualization Strategy


This report outlines an exploratory data analysis (EDA) and visualization strategy for uncovering patterns and anomalies in the accident data. It details the types of visualizations planned — including histograms, box plots, and scatter plots using matplotlib and seaborn — and explains how each supports specific analytical objectives, such as identifying relationships between weather conditions, road features, and accident severity. The document also discusses handling categorical sparsity in rare weather types and provides a plan for documenting and interpreting EDA findings.

### Week 4: Machine Learning Model Selection and Evaluation Plan

This report presents a complete Machine Learning Model Selection and Evaluation Plan for the Road Accident Severity Prediction project, using the US Accidents (Kaggle) dataset, and builds directly on the project planning, data cleaning, and exploratory analysis work completed in Weeks 1–3.

The report frames accident severity prediction as a three-class classification problem (Minor, Serious, Fatal) and evaluates five candidate algorithms — Logistic Regression, Decision Tree, Random Forest, Gradient Boosted Trees (XGBoost/LightGBM), and a Multi-Layer Perceptron — against explicit selection criteria including how well each handles mixed data types, class imbalance, non-linear feature interactions, interpretability, and training speed at scale. A comparison table summarizes these trade-offs, with LightGBM/XGBoost recommended as the primary candidate and Logistic Regression retained as an interpretable baseline.

To ground the theory in practice, the report includes worked numerical examples: a manual logistic-regression probability calculation for a sample record, and a full confusion-matrix walkthrough showing exactly how precision, recall, F1-score, and Macro-F1 are computed and why Macro-F1 is a more reliable model-selection metric than raw accuracy for this imbalanced dataset. A dedicated section addresses anticipated challenges — class imbalance in Fatal-severity cases, categorical sparsity in rare weather types, overfitting, and interpretability of ensemble models — with concrete mitigation strategies such as SMOTE, class weighting, category grouping/target encoding, and SHAP-based explanations.

The plan also specifies a stratified train/validation/test split, stratified k-fold cross-validation, and a two-stage hyperparameter search (RandomizedSearchCV followed by GridSearchCV), illustrated with an example LightGBM parameter grid. An end-to-end process flowchart visualizes the full pipeline, and a phased execution plan outlines how the strategy would be implemented in practice.

---

## 🛠️ Tools & Libraries

- **Data handling:** pandas, numpy
- **Modeling:** scikit-learn, xgboost, lightgbm
- **Imbalanced data:** imbalanced-learn (SMOTE)
- **Interpretability:** shap
- **Visualization:** matplotlib, seaborn
- **Hyperparameter tuning:** GridSearchCV, RandomizedSearchCV, optuna (optional)

---

## ✅ Status

| Week | Task | Status |
|------|------|--------|
| 1 | Project Planning and Dataset Scoping | ✅ Complete |
| 2 | Data Cleaning and Transformation Documentation | ✅ Complete |
| 3 | Exploratory Data Analysis and Visualization Strategy | ✅ Complete |
| 4 | ML Model Selection and Evaluation Plan | ✅ Complete |

---

## 👤 Author

Virtual Data Science Apprentice – Python Specialist Intern
