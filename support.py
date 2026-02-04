import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download stopwords (run once)
nltk.download('stopwords')

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\sanka\Desktop\FUTURE_ML_02\customer_support_tickets.csv")  # update path if needed

# Keep only required columns
df = df[['Ticket Description', 'Ticket Type']]
# Rename columns for easier access
df.columns = ['ticket_text', 'category']

# -----------------------------
# 2. Text Cleaning Function
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['ticket_text'].apply(clean_text)

# -----------------------------
# 3. Feature Extraction (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])

y = df['category']

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# 7. Priority Assignment Logic
# -----------------------------
def assign_priority(text):
    text = text.lower()
    if any(word in text for word in ['urgent', 'failed', 'error', 'down']):
        return 'High'
    elif any(word in text for word in ['delay', 'slow', 'issue']):
        return 'Medium'
    else:
        return 'Low'

df['Priority'] = df['ticket_text'].apply(assign_priority)

print("\nSample Output:")
print(df[['ticket_text', 'category', 'Priority']].head())
