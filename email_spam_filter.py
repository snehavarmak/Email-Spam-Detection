# importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# creating a small dataset manually
data = {
    'text': [
        "Congratulations! You have won a free ticket. Click here to claim.",
        "Hey, are you coming to class today?",
        "Win a brand new iPhone now, click the link!",
        "Please submit the assignment by tomorrow.",
        "You won a $500 gift card, claim your prize now!",
        "Let's meet for lunch at 2 pm.",
        "Get rich fast! Earn money from home!",
        "Your meeting is scheduled at 5 PM.",
        "Free vacation offer, click here now!",
        "Can you send me your project file?"
    ],
    'label': ['spam','ham','spam','ham','spam','ham','spam','ham','spam','ham']
}

# converting into dataframe
df = pd.DataFrame(data)

# splitting the dataset into training and testing data
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=0)

# converting text into numerical form using CountVectorizer
cv = CountVectorizer()
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)

# creating and training the model
model = MultinomialNB()
model.fit(x_train_cv, y_train)

# making predictions
y_pred = model.predict(x_test_cv)

# checking accuracy and performance
print("Accuracy of the model:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# testing with a custom message
sample_mail = ["You have won a lottery! Click here to claim now!"]
sample_mail_cv = cv.transform(sample_mail)
pred = model.predict(sample_mail_cv)

print("\nCustom Mail Test:")
print(sample_mail[0], "->", pred[0])
