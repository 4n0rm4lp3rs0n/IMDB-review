import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

data = pd.read_csv("IMDB Dataset.csv")
# data.head()

data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

def clearHTML(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
def clearpunc(text):
    return re.sub(r'[^\w\s]','',text)
def text_purify(text):
    text = clearHTML(text)
    text = clearpunc(text)
    text = text.lower()
    return text

data['review'] = data['review'].apply(text_purify)

#Convert text to nums
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['review'])
y = data['sentiment']

#Test train split and model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = BernoulliNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print()
print(classification_report(y_test, y_pred))

#Data saving
joblib.dump(model, 'IMDB.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')