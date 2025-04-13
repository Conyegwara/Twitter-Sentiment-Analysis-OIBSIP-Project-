import pandas as pd

df = pd.read_csv(r"C:\Users\onyeg\Downloads\pratice modules\dataset\twitter_data\Twitter_Data.csv")

print(df)

df = df.dropna()


df["category"] = df["category"].astype(int)

class_distribution = df["category"].value_counts(normalize=True) * 100
print(class_distribution)


df["category"] = df["category"].astype(int)

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df.dropna(inplace=True)

df['category'] = df['category'].astype(int)

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'\W', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['processed_text'] = df['clean_text'].astype(str).apply(preprocess_text)

vectorizer = TfidfVectorizer(max_features=5000)  
X = vectorizer.fit_transform(df['processed_text']).toarray()
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x=df['category'], palette="coolwarm")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.title("Sentiment Distribution in Dataset")
plt.xticks(ticks=[0, 1, 2], labels=["Negative", "Neutral", "Positive"])
plt.show()


