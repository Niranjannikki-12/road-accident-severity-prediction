This repository contains my work for the Virtual Data Science Apprentice Program — Python Specialist Intern (10-Aug-2026 to 07-Sep-2026). The project builds a Python-based data science pipeline to predict the severity of road accidents in the United States, using environmental, temporal, and road-condition features.

Dataset: US Accidents (2016–2023) — a countrywide traffic accident dataset compiled by Sobhan Moosavi et al., containing ~7.7 million records across 46 columns, hosted on Kaggle.

Contents
Week1_Project_Proposal_Road_Accident_Severity.docx — Week 1: project proposal covering objectives, scope, dataset selection, analytical workflow, methodology/tools, timeline, and a risk register.
Week2_Data_Cleaning_Transformation.docx — Week 2: data cleaning and transformation strategy, with worked calculations for missing-value handling, duplicate detection, outlier capping (IQR), and feature scaling/engineering.
cleaning_pipeline.py — Python implementation of the Week 2 cleaning strategy: missingness audit, tiered imputation, duplicate removal, outlier capping, type standardization, weather-category consolidation, z-score standardization, and time-based feature engineering.
Week3_EDA_Visualization_Strategy.pdf — Week 3: exploratory data analysis and visualization strategy, defining a six-stage EDA workflow and detailing six visualization types (severity distribution, hourly trend, weather frequency, visibility-by-severity, correlation heatmap, temperature/humidity scatter), each tied to an interpretation goal and a Week 4 modeling action point.
eda_visualization.ipynb — Python implementation of the Week 3 EDA plan: generates all six charts referenced in the report using pandas, matplotlib, and seaborn, with accompanying summary statistics (class proportions, peak hour, severity rate by weather, median visibility by severity, correlation coefficients).
figures/ — chart images produced by eda_visualization.ipynb.
Usage
python cleaning_pipeline.py --input US_Accidents.csv --output cleaned_accidents.csv
jupyter nbconvert --to notebook --execute eda_visualization.ipynb
Status

Weeks 1–3 complete. Week 4 (model selection and evaluation) in progress.
