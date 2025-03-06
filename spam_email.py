import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset (assuming you have 'email_spam.csv')
# This dataset should have two columns: 'message' (email text) and 'label' (spam/ham)
emails = pd.read_csv('emails_spam.csv')

# Example of how the dataset should look like
# message, label
# "Free money! Claim now", spam
# "Meeting at 10am tomorrow", ham
print(emails.head())

# Step 2: Preprocess the text
# We will use CountVectorizer to convert the text to a matrix of token counts (bag of words)
vectorizer = CountVectorizer(stop_words='english', lowercase=True)

# Convert email text into feature vectors
X = vectorizer.fit_transform(emails['message'])

# Step 3: Convert labels into numeric format (spam = 1, ham = 0)
y = emails['label'].apply(lambda x: 1 if x == 'spam' else 0)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Step 6: Predict the test set
y_pred = classifier.predict(X_test)

# Step 7: Evaluate the classifier performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 8: Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Test the classifier with a sample email
sample_email = ["Free money! Get your cash prize here."]
sample_vectorized = vectorizer.transform(sample_email)
prediction = classifier.predict(sample_vectorized)

if prediction == 1:
    print("\nThe email is classified as: Spam")
else:
    print("\nThe email is classified as: Ham")
