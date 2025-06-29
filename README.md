
---

# Human Activity Recognition (HAR) Using Smartphone Accelerometer Data

## Overview

This project focuses on classifying six human activities using accelerometer data from the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).

All classification models were built using **decision trees** and tested on three different types of features:

1. **Raw Accelerometer Features** – Mean of `acc_x`, `acc_y`, `acc_z`
2. **TSFEL-Extracted Features** – Statistical and temporal domain features
3. **Provided Features** – Precomputed features from `X_train.txt` and `X_test.txt`

---

## Dataset

* Source: UCI HAR Dataset
* Activities: `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`
* Sensors Used: Accelerometer (x, y, z axes)
* Sampling Rate: 50 Hz
* Window Length: 128 readings per sample (\~2.56 seconds)

---

## Libraries Used

* `pandas`, `numpy` – Data handling and manipulation
* `matplotlib`, `seaborn` – Visualization
* `scikit-learn` – PCA, `DecisionTreeClassifier`, evaluation metrics
* `tsfel` – Feature extraction (statistical and temporal domains)
* `tqdm` – Progress display during extraction

---

## Methodology

### Data Preparation

* Loaded and combined accelerometer signals from the dataset
* Mapped activity labels to numeric classes
* Organized training and testing sets

### Feature Extraction

* **Raw Features**: Computed mean of `acc_x`, `acc_y`, and `acc_z` per sample
* **TSFEL Features**: Extracted statistical and temporal features using `tsfel`
* **Provided Features**: Used directly from dataset files (`X_train.txt`, `X_test.txt`)

### Dimensionality Reduction

* Applied PCA to all three feature sets
* Plotted 2D and 3D PCA projections to assess class separability

### Correlation Analysis

* Generated correlation matrices for each feature set
* Identified and noted highly correlated features (correlation > 0.9)

### Classification with Decision Trees

* Trained a `DecisionTreeClassifier` on each feature type
* Evaluated models using:

  * Accuracy
  * Precision
  * Recall
  * Confusion Matrix
* Tested multiple tree depths and plotted accuracy vs depth

---

## Results

| Feature Type      | Accuracy | Precision | Recall   |
| ----------------- | -------- | --------- | -------- |
| Raw Accelerometer | Moderate | Moderate  | Moderate |
| TSFEL Features    | High     | High      | High     |
| Provided Features | Good     | Good      | Good     |

* **TSFEL features** produced the best results due to detailed statistical and temporal information
* **Raw features** were less effective due to lower dimensionality and high noise
* **Tree depth** significantly influenced performance and overfitting

---

## Visualizations

* Accelerometer waveforms per activity
* Total acceleration magnitude plots
* 3D trajectories of body movement
* PCA scatter plots (2D and 3D)
* Correlation heatmaps
* Confusion matrices
* Accuracy vs Tree Depth plots

---

## File Structure

```
├── data/          # Raw and processed accelerometer data
├── features/      # Extracted TSFEL features and provided features
├── analysis/      # Plots: PCA, correlation, confusion matrices
├── models/        # Decision tree training and evaluation scripts
├── scripts/       # Preprocessing and utility scripts (e.g., CombineScript.py)
├── README.md
```

---

## Notes

* All models were trained and tested **only using decision trees**
* TSFEL was configured to extract **statistical** and **temporal** features only
* Evaluation was consistent across all feature sets for fair comparison
* PCA and correlation analysis were used to better understand feature behavior

---

