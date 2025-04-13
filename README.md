# Twitter-Sentiment-Analysis-OIBSIP-Project-
### Description
---
This project focuses on building a sentiment analysis model to classify the emotional tone of Twitter data as positive, negative, or neutral. Leveraging Natural Language Processing (NLP) and Machine Learning (ML) techniques, it also demonstrates a complete sentiment analysis pipeline from data preprocessing to model training and evaluation, providing valuable insights into large-scale social media content.

### 📌 Overview
---
The main objective of this project is to build a sentiment analysis model capable of accurately identifying the emotional tone of textual data. By leveraging Natural Language Processing (NLP) and Machine Learning (ML) techniques, the model aims to classify tweets into three sentiment categories: positive, neutral, and negative. This helps extract meaningful insights from large volumes of public opinion, customer feedback, and social media discussions. The dataset used comprises over 160,000 tweets, primarily centered around political topics and broader social discourse.

### Project Objectives

- Analyze and classify public sentiment on Twitter.
- Implement machine learning models to predict sentiment based on tweet content.
- Visualize sentiment distribution and model performance for clear interpretation.
- Document and demonstrate a reproducible NLP pipeline

### 🧰 Tools & Libraries

- Programming Language: **Python**
- Data Handling: **pandas, numpy**
- Text Processing: **re, nltk**
- Feature Extraction: **sklearn.feature_extraction.text.TfidfVectorizer**
- Modeling: **MultinomialNB from sklearn.naive_bayes**
- Evaluation: **classification_report, confusion_matrix**
- Visualization: **matplotlib, seaborn**

### 📂 Data Overview
 - <a href="https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset">Data source</a>

 - <a href="https://github.com/Conyegwara/Twitter-Sentiment-Analysis-OIBSIP-Project-/blob/main/Twitter_Data.csv">Dataset</a>

- Columns:
  - clean_text: Cleaned text of the tweet
  - category: Sentiment label
    - 1.0 → Positive
    - 0.0 → Neutral
    - -1.0 → Negative
       
- Size: ~163,000 rows

#### 🔄 Data Preprocessing

-1. Missing Values: Removed all rows with null values in either clean_text or category
  - $df.dropna(inplace=True)$
 
- Text Cleaning:
  - Lowercasing
  - Removing URLs, mentions, hashtags, punctuation
  - Tokenizing, removing stopwords, and lemmatization using nltk
    - $def clean_text(text)$:
    - $...$
    - $return ' '.join(tokens)$
   










