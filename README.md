# Email-Spam-Detection

# 📧 Email Spam Filter using Machine Learning

### 🧠 Overview

This project is a simple **Email Spam Detection System** built using **Python** and **Machine Learning**. It classifies emails as **Spam** or **Ham (Not Spam)** using a small dataset and a **Naïve Bayes** algorithm.


### ⚙️ Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn (sklearn)**


### 🚀 Features

* Manually created small dataset of sample emails
* Converts text data into numerical features using **CountVectorizer**
* Uses **Multinomial Naïve Bayes** classifier
* Splits data into training and testing sets
* Displays model **accuracy** and **classification report**


### 🧩 How It Works

1. Load and prepare sample email text data
2. Convert text to numerical form using **CountVectorizer**
3. Train model with **MultinomialNB**
4. Evaluate model performance using **accuracy_score** and **confusion_matrix**


### 📊 Example Output

Accuracy: 0.90
Confusion Matrix:
[[3 0]
 [1 3]]
Classification Report:
              precision    recall  f1-score   support
         ham       0.75      1.00      0.86         3
        spam       1.00      0.75      0.86         4

### 👩‍💻 Author

**Sneha Latha Kollu**
