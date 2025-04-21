#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ------------------------------------
# 🔹 STEP 1: LOAD DATASET FROM CSV
# ------------------------------------
csv_path = r"C:\Users\pavin\OneDrive\Desktop\combined_emails_with_natural_pii.csv"  # Replace with your actual file path
df = pd.read_csv(csv_path)

# ------------------------------------
# 🔹 STEP 2: AUTOMATICALLY DETECT LABELS
# ------------------------------------
# Check the unique labels in the 'type' column (detects all labels used)
labels = df['type'].unique()
print(f"Detected labels: {labels}")

# Encoding labels into numerical values using LabelEncoder
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['type'])

# ------------------------------------
# 🔹 STEP 3: PREPROCESSING TEXT DATA
# ------------------------------------
# Use TfidfVectorizer to convert emails to feature vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # You can tweak max_features

# Vectorize the email text data
X = vectorizer.fit_transform(df['email']).toarray()  # X is the feature matrix (emails converted to TF-IDF features)
y = df['label']  # y is the label column (encoded labels)

# ------------------------------------
# 🔹 STEP 4: TRAIN/TEST SPLIT
# ------------------------------------
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------------
# 🔹 STEP 5: TRAIN A CLASSICAL ML MODEL
# ------------------------------------
# Use Logistic Regression as the classifier
classifier = LogisticRegression(max_iter=1000)  # Increase max_iter if necessary
classifier.fit(X_train, y_train)

# ------------------------------------
# 🔹 STEP 6: EVALUATE THE MODEL
# ------------------------------------
# Predict on the test set
y_pred = classifier.predict(X_test)

# Print classification report (precision, recall, F1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ------------------------------------
# 🔹 STEP 7: PREDICTING ON NEW EMAILS
# ------------------------------------
# Function to predict the label of a new email
def predict_email(email):
    # Vectorize the email
    email_vec = vectorizer.transform([email]).toarray()
    
    # Predict the label (0 = 'problem', 1 = 'request', etc.)
    predicted_label = classifier.predict(email_vec)[0]
    
    # Convert the numerical label back to the original string label
    original_label = label_encoder.inverse_transform([predicted_label])[0]
    return original_label

# Example: Predict a new email
new_email = "I am facing issues with my order. Can you assist?"
predicted_label = predict_email(new_email)
print(f"Predicted label for the new email: {predicted_label}")


# In[13]:





# In[10]:





# In[15]:





# In[ ]:





# In[ ]:




