# Twitter-Sentiment-Analysis-OIBSIP-Project-
## Table Of Content

- [Description](#description)
- [📌Project Overview](#📌project-overview)
- [Project Objectives](#project-objectives)
- [🧰Tools & Libraries](#🧰tools-&-libraries)


### Description
---
This project focuses on building a sentiment analysis model to classify the emotional tone of Twitter data as positive, negative, or neutral. Leveraging Natural Language Processing (NLP) and Machine Learning (ML) techniques, it also demonstrates a complete sentiment analysis pipeline from data preprocessing to model training and evaluation, providing valuable insights into large-scale social media content.

### 📌Project Overview
---
The main objective of this project is to build a sentiment analysis model capable of accurately identifying the emotional tone of textual data. By leveraging Natural Language Processing (NLP) and Machine Learning (ML) techniques, the model aims to classify tweets into three sentiment categories: positive, neutral, and negative. This helps extract meaningful insights from large volumes of public opinion, customer feedback, and social media discussions. The dataset used comprises over 160,000 tweets, primarily centered around political topics and broader social discourse.

![confusion matrix](https://github.com/user-attachments/assets/0b5289f7-2783-4b18-a8c7-9d56594c9ca0)


### Project Objectives

- Analyze and classify public sentiment on Twitter.
- Implement machine learning models to predict sentiment based on tweet content.
- Visualize sentiment distribution and model performance for clear interpretation.
- Document and demonstrate a reproducible NLP pipeline

### 🧰Tools & Libraries

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

1. **Missing Values**: Removed all rows with null values in either clean_text or category
  - $df.dropna(inplace=True)$
 
 2. **Text Cleaning:**
  - Lowercasing
  - Removing URLs, mentions, hashtags, punctuation
  - Tokenizing, removing stopwords, and lemmatization using nltk
    - $def clean_text(text)$:
    - $...$
    - $return ' '.join(tokens)$
   
3. **TF-IDF Feature Engineering:**
  - Text was transformed into numerical vectors using TF-IDF with 5000 max features.
  - $vectorizer = TfidfVectorizer(max_features=5000)$
  - $X = vectorizer.fit_transform(df['clean_text'])$

### 🤖 Model Training

- Algorithm: Multinomial Naive Bayes (MultinomialNB)
- Train-Test Split: 80% training / 20% testing
  - $model = MultinomialNB()$
  - $model.fit(X_train, y_train)$


### 📊 Evaluation Metrics

- Model accurancy: 73%
-  Setiment Average
   - positive = 44%
   - Neutral = 34%
   -  Negative = 22%
  
- Classification Report

|Sentiment|Precision|Recall|F1-score|support|
|---------|---------|------|--------|-------|
|Negative(-1.0)|88%|44%|58%|7152|
|Neutral(0.0)|84%|66%|74%|11067|
|Positive(1.0)|65%|93%|77%|14375|
|Accurancy|-|-|73%|32594|
|Macro Avg|79%|67%|69%|32594|
|Weight Avg|77%|73%|72%|32594|

### Observations

- Excellent precision for negative tweets, but recall is low → model is cautious with negatives.
- Positive tweets are well-detected, high recall.
- Neutral class performs consistently.

### 📈 Visualizations
- 🔹 Confusion Matrix
- 🔹 Sentiment Distribution
- A bar chart showing the original distribution of sentiment labels in the dataset.

  ![sentiment distribution](https://github.com/user-attachments/assets/f0b710fa-2007-484d-8227-4eb56f4f4cd1)


### ✅ Conclusions
- TF-IDF + Naive Bayes is a strong baseline, achieving ~73% accuracy.

- The model generalizes well for positive/neutral classes; improvement is needed for negative tweets.

- Advanced techniques like SVM, logistic regression, or BERT could further improve performance.

### 🚀 Future Improvements

- Try deep learning (e.g., LSTM, BERT)

- Use more sophisticated text embeddings (Word2Vec, GloVe, BERT)

- Handle sarcasm, slang, and multilingual tweets

- Deploy the model using Flask/Streamlit for real-time sentiment classification

#### Note
check the files section for the python code and solution































