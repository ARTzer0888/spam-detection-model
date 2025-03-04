# spam-detection-model

*COMPANY*: CODTECH ID SOLUTIONS

*NAME*: RAJ SHANKAR PATIL

*INTERN ID*: CODHC215

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

# Spam Detection Model

## Project Overview

This project demonstrates the creation of a predictive model using scikit-learn to classify messages as **Spam** or **Ham** (non-spam). The model is trained using a simple dataset containing various types of messages, some of which are spam and some are ham. The classifier used in this model is **Naïve Bayes**.

## Task

The task is to **create a predictive model** to classify or predict outcomes (in this case, whether an email is spam or ham) from a dataset. We will use **scikit-learn**'s **Multinomial Naïve Bayes** classifier and evaluate its performance on a test set.

## Dataset

The dataset consists of pairs of messages and their corresponding labels (`ham` or `spam`):

- `ham`: Legitimate message (non-spam)
- `spam`: Unsolicited or potentially harmful message (spam)

### Example Messages

- Ham: "Hey, are we still on for dinner tonight?"
- Spam: "Congratulations! You've won a free iPhone! Click the link to claim."

## Model Description

The model is trained on the dataset using the following steps:

1. **Preprocessing**: The messages are converted to numerical features using **TF-IDF** (Term Frequency-Inverse Document Frequency) vectorization, with n-grams considered (1-gram and 2-gram).
2. **Model Training**: The **Multinomial Naive Bayes** classifier is used to train the model.
3. **Evaluation**: The model is evaluated using metrics like **accuracy**, **classification report**, and **confusion matrix**.
4. **Testing**: The model is tested on new unseen messages to predict if they are spam or ham.

## Dependencies

- `pandas`
- `sklearn`
- `numpy`

## Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spam-detection-model.git
     ```

#OUTPUT
```bash
Model Accuracy: 71.43%

Classification Report:
               precision    recall  f1-score   support

           0       0.60      1.00      0.75         3
           1       1.00      0.50      0.67         4

    accuracy                           0.71         7
   macro avg       0.80      0.75      0.71         7
weighted avg       0.83      0.71      0.70         7


Confusion Matrix:
 [[3 0]
 [2 2]]
Message: 'Congratulations! You've won a $1000 gift card. Click here to claim.' => Spam
Message: 'Limited-time offer! Get 70% off on all products. Don't miss out!.' => Spam
Message: 'Hey, do you have time to talk?' => Ham
Message: 'Your PayPal account has been suspended. Verify immediately!' => Spam
Message: 'Reminder: Your appointment is scheduled for tomorrow.' => Ham
Message: 'Limited-time deal! Get 50% off on all electronics.' => Spam
   
