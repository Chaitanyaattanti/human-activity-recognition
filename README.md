# Human Activity Recognition (HAR) Using Smartphone Accelerometer Data

## Overview

This project focuses on recognizing six common human activities using smartphone-based accelerometer data. The goal is to build reliable models using both classical machine learning (decision trees) and deep learning (neural networks). We used features from raw signals, TSFEL, and the dataset's provided features. Augmentation and self-recorded data were also added to improve model performance.

---

## Dataset

* **Source**: [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
* **Participants**: 30 people performing 6 activities
* **Activities**:

  * WALKING
  * WALKING\_UPSTAIRS
  * WALKING\_DOWNSTAIRS
  * SITTING
  * STANDING
  * LAYING
* **Sensors**: Accelerometer and gyroscope (x, y, z axes)
* **Sampling Rate**: 50 Hz
* **Window Size**: 128 readings (\~2.56 seconds)
* **Additional Data**: Extra accelerometer recordings were collected and added to increase robustness

---

## Project Structure

```
├── data/          # Raw and processed signal data
├── features/      # TSFEL and provided features
├── models/        # ML and NN training code
├── analysis/      # Plots and evaluation outputs
├── scripts/       # Preprocessing and utility functions
├── README.md
```

---

## Methodology

### 1. Data Preparation

* Combined raw signals (`acc_x`, `acc_y`, `acc_z`) into samples
* Applied windowing with offset to create stable segments
* Used stratified train-test split to balance activity labels

### 2. Feature Extraction

* **Raw features**: Mean of `acc_x`, `acc_y`, `acc_z`
* **TSFEL features**: Extracted statistical and temporal features using the TSFEL library
* **Provided features**: Precomputed features from the dataset (`X_train.txt`, `X_test.txt`)

### 3. Visualization & Analysis

* **PCA** was used to reduce dimensionality and visualize separability
* **Correlation heatmaps** helped identify redundant features

### 4. Models Used

* **Decision Tree (Scikit-learn)**: Trained and evaluated on all three feature types
* **Custom Decision Tree**: Implemented from scratch using NumPy and Pandas
* **Neural Network (PyTorch)**: Trained on TSFEL and augmented data
* **Scratch Neural Network**: Basic NN written from scratch for learning

### 5. Data Augmentation

* Used AugLy to create new samples (e.g., with noise or time shifts)
* Combined with self-collected data to make models more robust

---

## Results
| Model                      | Feature Type           | Accuracy             |
| -------------------------- | ---------------------- | -------------------- |
| Decision Tree              | TSFEL Features         | **93%**              |
| Decision Tree (Scratch)    | Time-Domain Features   | **91%**              |
| Neural Network (PyTorch)   | TSFEL Features         | **97%**              |
| Neural Network (Scratch)   | TSFEL Features         | **95%**              |
| Neural Network (Augmented) | TSFEL + Augmented Data | **87% (mixed-sample)** |
| Decision Tree              | Provided Features      | \~85–88%             |
| Decision Tree              | Raw Accelerometer      | \~70–75%             |


* TSFEL features gave the best overall performance
* Neural networks performed better than decision trees, especially with augmentation
* Raw features were the least effective due to low information content

---

## Visualizations

* Raw signal plots for each activity
* Acceleration magnitude trends
* 2D and 3D PCA scatter plots
* Correlation heatmaps
* Confusion matrices
* Accuracy vs decision tree depth plots

---

## Future Work

* Deploy models on mobile or wearable devices
* Add more activity types or transitions
* Try RNNs or Transformers for sequential modeling
* Analyze misclassifications and edge cases

---

## Acknowledgements

* UCI HAR Dataset
* Libraries used: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, PyTorch, TSFEL, AugLy

---

---

