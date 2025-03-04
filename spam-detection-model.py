import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


messages = [
    ("ham", "Hey, are we still on for dinner tonight?"),
    ("spam", "Congratulations! You've won a free iPhone! Click the link to claim."),
    ("ham", "Don't forget to submit your assignment by tomorrow."),
    ("spam", "Urgent! Your account has been compromised. Verify now."),
    ("ham", "Looking forward to our meeting next week."),
    ("spam", "You have been selected for a $500 gift card! Claim it now."),
    ("spam", "Final notice! Your car warranty is expiring soon."),
    ("ham", "Can you send me the report by today?"),
    ("spam", "Click here to get 70% off on all products!"),
    ("ham", "Are you coming to the party tonight?"),
    ("spam", "Limited-time offer! Win a vacation by signing up."),
    ("ham", "Let's catch up for lunch."),
    ("spam", "Your bank account has been locked due to suspicious activity."),
    ("ham", "Meeting at 3 PM. Don't be late."),
    ("ham", "Let's schedule a call for tomorrow."),
    ("ham", "Don't forget about the team meeting at 10 AM."),
    ("ham", "How was your weekend?"),
    ("ham", "Can you help me with this issue?"),
    ("ham", "Looking forward to our trip next month."),
    ("spam", "Hurry! Claim your prize before it's too late."),
    ("spam", "Act now! Limited-time investment opportunity."),
    ("spam", "You won a free lottery ticket! Sign up now.")
]

# Convert data into a DataFrame
df = pd.DataFrame(messages, columns=["label", "message"])

# Encode labels (0 for ham, 1 for spam)
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(df["message"], df["label"], test_size=0.3, random_state=42)

# Convert text into numerical features using TF-IDF with n-grams
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Naïve Bayes classifier with optimized smoothing (alpha=1)
model = MultinomialNB(alpha=1.0)  # Changed alpha to 1 for smoothing
model.fit(X_train_vec, y_train)

# Evaluate model performance
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred, zero_division=1)

# Print confusion matrix to understand errors
conf_matrix = confusion_matrix(y_test, y_pred)

# Test with new messages
test_messages = [
    "Congratulations! You've won a $1000 gift card. Click here to claim.",
    "Limited-time offer! Get 70% off on all products. Don't miss out!.",
    "Hey, do you have time to talk?",
    "Your PayPal account has been suspended. Verify immediately!",
    "Reminder: Your appointment is scheduled for tomorrow.",
    "Limited-time deal! Get 50% off on all electronics."
]

# Convert new messages to TF-IDF format and predict
test_messages_vec = vectorizer.transform(test_messages)
predictions = model.predict(test_messages_vec)

# Output predictions
print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)

for msg, pred in zip(test_messages, predictions):
    print(f"Message: '{msg}' => {'Spam' if pred == 1 else 'Ham'}")
