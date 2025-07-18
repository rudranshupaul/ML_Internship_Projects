# 🚲 Bike Sharing Demand Prediction

## Overview

Predicting daily bike rental counts using weather, calendar, and environmental data from the [UCI/Kaggle Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset). This project demonstrates all steps of a modern ML pipeline: EDA, feature engineering, regression modeling, model comparison, and final business insights.

## Dataset

- Source: [UCI Bike Sharing Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset)
- Data not included; download from the link above (`day.csv`).
- Citation required if used in publications (see "License" below).

## Features

- **Data Cleaning & EDA:** Explored trends across seasons, weather, holidays, and weekdays.
- **Feature Engineering:** Created new time-based and user features for modeling.
- **Modeling:** Compared several regression models (Random Forest, Gradient Boost, linear models).
- **Performance:** Achieved R² = 0.917 (Random Forest), RMSE ≈ 533.
- **Insights:** Demand is seasonal, peaks in summer/fall, higher for registered users, sensitive to weather.

## How to Run

1. Install dependencies:
    ```
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

2. Download `day.csv` from the [UCI dataset link](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset) and place it in the project folder.

3. Run:
    ```
    python bike_sharing_demand.py
    ```

4. Check results and plots in the output directory.

## Key Results

| Model              | R²     | RMSE  | MAE   |
|--------------------|--------|-------|-------|
| Random Forest      | 0.917  | 533   | 396   |
| HistGradientBoost  | 0.91   | 572   | 430   |
| Linear Regression  | 0.77   | 885   | 701   |

## Business Insights

- Rentals peak in summer and on working days.
- Temperature and weather strongly affect demand.
- Registered users form over 80% of rentals.

## License & Citation

Data source and terms:  
Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15.  
[Springer link](https://doi.org/10.1007/s13748-013-0040-3)

---

## About

Internship project by [Your Name], July 2025.  
See `/day.csv` for data structure (not distributed in this repo).

