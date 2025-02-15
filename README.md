# Parking Lot Detector

This script performs a simple classification task to distinguish whether a parking space is **empty** or **not empty** using a **Support Vector Machine (SVM)** and **Grid Search** for hyperparameter optimization.

## 1. Project Description

This project uses:
- **scikit-image** for reading and resizing images.
- **NumPy** for array and data manipulation.
- **scikit-learn** for splitting the dataset into training and testing sets, and for classification with **SVC** (Support Vector Classifier).

In essence, the project will:
1. Read images from two folders: `data/empty` and `data/not_empty`.
2. Resize each image to 15×15 pixels.
3. Flatten each image into a 1D vector for the SVM model.
4. Split the dataset into training and testing sets.
5. Use **GridSearchCV** to find the best hyperparameters.
6. Measure the **accuracy** of the best model on the test set.

## 2. Requirements (Dependencies)

Make sure you have the following libraries installed before running the script:

- Python 3.x
- NumPy
- scikit-image
- scikit-learn

You can install them via `pip`:
```bash
pip install numpy scikit-image scikit-learn
