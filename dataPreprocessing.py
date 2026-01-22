# dataPreprocessing.py

import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class SpamClassifier:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.ps = PorterStemmer()
        nltk.download('stopwords')
        self.stopwords_set = set(stopwords.words('english'))
        self.vectorizer = CountVectorizer()
        self.model = RandomForestClassifier(n_jobs=-1, random_state=42)

    def _preprocess(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        words = [self.ps.stem(w) for w in words if w not in self.stopwords_set]
        return ' '.join(words)

    def load_and_prepare_data(self):
        df = pd.read_csv(self.csv_path)
        df['clean_text'] = df['text'].apply(self._preprocess)
        return df

    def train(self):
        df = self.load_and_prepare_data()
        X = self.vectorizer.fit_transform(df['clean_text'])
        y = df['label_num']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def predict(self, email_text: str) -> int:
        processed = self._preprocess(email_text)
        vector = self.vectorizer.transform([processed])
        return int(self.model.predict(vector)[0])
