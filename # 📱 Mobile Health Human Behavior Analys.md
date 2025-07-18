# 📱 Mobile Health Human Behavior Analysis: Human Activity Recognition

## Overview

This project utilizes the MHEALTH (Mobile Health) dataset to classify human activities based on motion sensor signals. The dataset includes recordings of 12 common activities performed by 10 diverse volunteers, leveraging wearable sensors placed on the chest, right wrist, and left ankle. The objective is to use acceleration and gyroscope data to accurately recognize human behaviors in real-world, out-of-lab conditions.

---

## Dataset Summary

- **Activities:** 12
- **Subjects:** 10
- **Sensors:** 3 (Chest, Right Wrist, Left Ankle)
- **Sampling Rate:** 50 Hz
- **Features Used:** Acceleration and gyroscope data (ECG and magnetometer available in original dataset)
- **Data Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/MHEALTH+Dataset)
- **License:** CC0: Public Domain

---

## Activities Included

| Label | Activity                         | Duration/Reps  |
|-------|----------------------------------|---------------|
| L1    | Standing still                   | 1 minute      |
| L2    | Sitting and relaxing             | 1 minute      |
| L3    | Lying down                       | 1 minute      |
| L4    | Walking                          | 1 minute      |
| L5    | Climbing stairs                  | 1 minute      |
| L6    | Waist bends forward              | 20 times      |
| L7    | Frontal elevation of arms        | 20 times      |
| L8    | Knees bending (crouching)        | 20 times      |
| L9    | Cycling                          | 1 minute      |
| L10   | Jogging                          | 1 minute      |
| L11   | Running                          | 1 minute      |
| L12   | Jump, front & back               | 20 times      |

---

## Experimental Setup

- Wearable Shimmer2 sensors were placed on:
  - Chest (includes ECG, not used here)
  - Right-lower-arm (wrist)
  - Left-ankle
- Sensors measured acceleration (X, Y, Z), gyroscope rotation (X, Y, Z), and magnetic orientation (not used here).
- Data collected during everyday activities, with video recording for ground truth labeling.
- Data processed to include **only acceleration and gyroscope features** for activity recognition.

---

## Attribute Information

| Column | Description                                           |
|--------|------------------------------------------------------|
| alx    | Left ankle acceleration (X axis)                     |
| aly    | Left ankle acceleration (Y axis)                     |
| alz    | Left ankle acceleration (Z axis)                     |
| glx    | Left ankle gyro (X axis)                             |
| gly    | Left ankle gyro (Y axis)                             |
| glz    | Left ankle gyro (Z axis)                             |
| arx    | Right arm acceleration (X axis)                      |
| ary    | Right arm acceleration (Y axis)                      |
| arz    | Right arm acceleration (Z axis)                      |
| grx    | Right arm gyro (X axis)                              |
| gry    | Right arm gyro (Y axis)                              |
| grz    | Right arm gyro (Z axis)                              |
| subject| Volunteer identifier                                 |
| Activity | Activity label (L1-L12 as above)                   |

*For more on the original magnetometer and ECG channels, see the full dataset [here](https://archive.ics.uci.edu/ml/datasets/MHEALTH+Dataset).*

---

## Project Workflow

- **Data Loading & Preprocessing:** Selected only acceleration and gyroscope channels, handled missing values, and encoded activity labels for classification.
- **Exploratory Data Analysis:** Visualized class balance, feature distributions, and sensor correlation to explore informative input signals.
- **Feature Engineering:** Created combined activity instances using multi-sensor data for robust recognition.
- **Model Building:** Compared supervised learning models (Random Forest, Gradient Boosting, Decision Tree, Logistic Regression, Naive Bayes) for multiclass classification.
- **Model Evaluation:** Used stratified train-test split, accuracy, precision, recall, and F1-score for in-depth performance reporting.
- **Result Highlights:** Achieved best F1-score of 0.94+ (Random Forest) on a challenging, real-world activity recognition problem.

---

## How to Run

1. **Install dependencies**  
#pip install pandas numpy scikit-learn matplotlib seaborn
2. **Download the dataset:**  
Download the processed [mhealth_raw_data.csv](https://www.kaggle.com/datasets/gaurav2022/mobile-health)
3. **Place the data file** in the project folder.
4. **Run the script:**  
python human_activity_detection.py
5. **Review results:**  
Outputs will include printed performance metrics and confusion matrix plots.

---

## Results

| Model              | Accuracy | F1-Score |
|--------------------|----------|----------|
| Random Forest      | 0.94+    | 0.94+    |
| HistGradientBoost  | 0.92     | 0.92     |
| Logistic Regression| 0.88     | 0.88     |
| Others             | See script for details |

---

## Acknowledgements

- Data from the [UCI Machine Learning Repository: MHEALTH Dataset](https://archive.ics.uci.edu/ml/datasets/MHEALTH+Dataset)
- Data preprocessing idea adapted from [Kaggle notebook](https://www.kaggle.com/gaurav2022/download-data-from-ucs/output)
- Internship project, July 2025

---

## License

CC0: Public Domain — See [dataset details](https://archive.ics.uci.edu/ml/datasets/MHEALTH+Dataset).

---

**Contact:** Rudranshu Paul  
*For more details or questions, open an issue or contact via GitHub.*
