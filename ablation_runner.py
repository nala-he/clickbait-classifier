
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# Ablation Runner
# This module runs a single ablation experiment with specified configurations
# to evaluate the impact of preprocessing steps on model performance. 
def run_ablation_experiment(data, vectorizer_type='tfidf', use_stopwords=True, truncate=False):
    # Prepare headline text
    if truncate:
        texts = data['headline'].apply(lambda x: ' '.join(x.split()[:5]))
    else:
        texts = data['headline']

    # Choose stopword setting
    stop_words = 'english' if use_stopwords else None

    # Choose vectorizer
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 2), max_features=5000)
    elif vectorizer_type == 'bow':
        vectorizer = CountVectorizer(stop_words=stop_words, ngram_range=(1, 2), max_features=5000)
    else:
        raise ValueError("vectorizer_type must be 'tfidf' or 'bow'")

    # Feature extraction
    X = vectorizer.fit_transform(texts)
    y = data['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train and evaluate model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Report results + return model and vectorizer
    return {
        'Vectorizer Type': vectorizer_type,
        'Stopwords': 'Removed' if use_stopwords else 'Kept',
        'Truncated': 'Yes' if truncate else 'No',
        'F1 Score': round(f1_score(y_test, y_pred, average='macro'), 4),
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Trained Model': model,
        'Trained Vectorizer': vectorizer
    }
