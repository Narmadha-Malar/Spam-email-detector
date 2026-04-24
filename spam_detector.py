# ============================================
# SPAM EMAIL DETECTOR - AI/ML Project
# By: Narmadha Malar K
# Tools: Python, Scikit-learn, Pandas
# ============================================

# STEP 1: Import Libraries
# Libraries = ready-made tools we can use directly

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("Libraries loaded!")

# ============================================
# STEP 2: Create Email Data
# emails = list of email messages
# labels = 1 means Spam, 0 means Not Spam
# ============================================

emails = [
    "Congratulations! You won 1 crore lottery click here now",
    "Free iPhone winner claim your prize immediately",
    "Buy cheap medicines online no prescription needed",
    "You are selected for cash reward call now",
    "Win big money fast easy click link below",
    "Hi Narmadha, can we meet tomorrow for project discussion?",
    "Your electricity bill is due please pay by Friday",
    "Mom asked me to remind you about dinner tonight",
    "Please find the attached resume for your review",
    "Team meeting scheduled for Monday at 10 AM",
    "Get rich quick scheme join today limited offer",
    "Urgent your account will be closed click verify",
    "Happy birthday! Hope you have a wonderful day",
    "Your order has been shipped tracking number 12345",
    "Make money from home no experience needed",
]

# 1 = Spam, 0 = Not Spam
labels = [1, 1, 1, 1, 1,
          0, 0, 0, 0, 0,
          1, 1, 0, 0, 1]

print(f"Total emails: {len(emails)}")
print(f"Spam emails: {labels.count(1)}")
print(f"Real emails: {labels.count(0)}")

# ============================================
# STEP 3: Split Data into Train and Test
# 80% data = model will learn from this
# 20% data = we will test the model on this
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    emails,
    labels,
    test_size=0.2,     # 20% for testing
    random_state=42    # fixed value so result is same every time
)

print(f"Training emails: {len(X_train)}")
print(f"Testing emails: {len(X_test)}")

# ============================================
# STEP 4: Convert Text to Numbers
# Computer cannot understand words
# So we convert words into numbers
# CountVectorizer counts how many times each word appears
# ============================================

vectorizer = CountVectorizer()

# Learn from training data and convert to numbers
X_train_vec = vectorizer.fit_transform(X_train)

# Convert test data to numbers using same rules
X_test_vec = vectorizer.transform(X_test)

print(f"Total unique words found: {len(vectorizer.vocabulary_)}")

# ============================================
# STEP 5: Train the Model
# We use Naive Bayes algorithm
# It is simple and works very well for text
# Model will learn which words are spam words
# ============================================

model = MultinomialNB()

# Train the model using training data
model.fit(X_train_vec, y_train)

print("Model training done!")

# ============================================
# STEP 6: Check Accuracy
# We test on unseen emails to check how good our model is
# ============================================

y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.1f}%")

# ============================================
# STEP 7: Test with New Emails
# Give any email and model will say spam or not
# ============================================

def check_email(email_text):
    # Convert email to numbers
    email_vec = vectorizer.transform([email_text])

    # Ask model to predict
    prediction = model.predict(email_vec)[0]

    # Return result
    if prediction == 1:
        return "SPAM - Do not open this email!"
    else:
        return "NOT SPAM - This email is safe!"

# Test with new emails
print("\n--- Testing New Emails ---")

test_emails = [
    "You won free money click here now",
    "Hi can we schedule a meeting tomorrow",
    "Claim your lottery prize today urgent",
    "Please review the attached document",
]

for email in test_emails:
    result = check_email(email)
    print(f"Email: {email}")
    print(f"Result: {result}")
    print()

print("Project Complete!")
