# Machine Learning Internship Tasks

This repository contains two machine learning projects that I completed as part of my internship.  
Both tasks focus on building classification models using real datasets.

# Task 1: Email Spam Classification

In this task, I built a model to classify emails as spam or not spam.

# What I did:
- Loaded and explored the dataset
- Cleaned the data (removed duplicates)
- Preprocessed the text (lowercase, removed punctuation, stopwords, etc.)
- Converted text into numerical features using CountVectorizer
- Split the dataset into training and testing sets
- Trained a Naive Bayes model
- Evaluated the model using accuracy and other metrics

# Result:
The model achieved around **97% accuracy** on the test data.

This task helped me understand text preprocessing, feature extraction, and supervised classification.

# Task 2: MNIST Digit Recognition

In this task, I built a model to recognize handwritten digits (0–9) using the MNIST dataset.

# What I did:
- Loaded and visualized the MNIST dataset (28×28 images)
- Normalized pixel values
- Built a Convolutional Neural Network (CNN)
- Trained the model for multiple epochs
- Evaluated the model on test data

# Result:
The model achieved around **98% test accuracy**.

This task helped me understand image preprocessing and how CNNs work for image classification.

# Technologies Used
- Python
- Scikit-learn

# Task 3: Housing Price Prediction

## Overview

In this task, I built a machine learning model to predict house prices using the California Housing dataset. The goal was to understand how different features like income, number of rooms, and location affect housing prices.

---

## What I Did

* Loaded the California housing dataset using sklearn
* Explored the dataset and checked for missing values
* Selected relevant features and target variable
* Split the data into training and testing sets
* Applied feature scaling
* Trained a Linear Regression model
* Evaluated the model using MSE and R² score
* Improved the model using Random Forest

---

## Models Used

* Linear Regression
* Random Forest Regressor

---

## Results

* Linear Regression gave moderate accuracy (R² ≈ 0.6)
* Random Forest performed better (R² ≈ 0.8)

---

## Conclusion

The model was able to predict house prices with reasonable accuracy.
Random Forest performed better than Linear Regression, showing that more advanced models can capture patterns in the data more effectively.

---

## Tools & Libraries

* Python
* Pandas, NumPy
* Scikit-learn

---

- TensorFlow / Keras
- NumPy
- Matplotlib

These projects demonstrate my understanding of supervised learning, preprocessing, model training, and evaluation.


# Task 4: Iris Flower Classification

## Overview

In this task, I built a machine learning model to classify iris flowers into different species based on their features like sepal length, sepal width, petal length, and petal width.

---

## What I Did

* Loaded the Iris dataset using sklearn
* Checked the dataset for missing values
* Selected features and target variable
* Split the data into training and testing sets
* Applied feature scaling
* Trained a Logistic Regression model
* Evaluated the model using accuracy

---

## Model Used

* Logistic Regression

---

## Results

* The model achieved an accuracy of **1.0 (100%)** on the test data

---

## Conclusion

The model performed very well and was able to classify iris flowers correctly.
Since the dataset is simple and well-structured, Logistic Regression was able to achieve very high accuracy.

---

## Tools & Libraries

* Python
* Pandas, NumPy
* Scikit-learn

---
