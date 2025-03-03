import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def read_corpus_sentiment(filename):
    with open(filename) as file:
        corpus = []
        y = []
        for line in file:
            sentence, sentiment = line.strip().rsplit('\t', 1)
            corpus.append(sentence)
            y.append(int(sentiment))
        return corpus, y

corpus_train, y_train = read_corpus_sentiment('yelp_labelled_train.txt')
corpus_test, y_test = read_corpus_sentiment('yelp_labelled_test.txt')

# TfidfVectorization
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
X_train = vectorizer.fit_transform(corpus_train)
X_test = vectorizer.transform(corpus_test)

# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
correct_classified = sum(true == pred for true, pred in zip(y_test, y_pred))
total_classified = len(y_test)
accuracy = correct_classified / total_classified
error_rate = 1-accuracy
print(error_rate)

# Save the model and vectorizer
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
