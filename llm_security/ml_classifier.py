"""
A real, trained scikit-learn classifier for prompt injection detection —
TF-IDF vectorization + Logistic Regression. This is the "catches
variations the heuristics miss" layer: the pattern-based detector in
prompt_injection_detector.py only catches phrasings it explicitly
matches; a trained classifier can generalize to reworded attacks that
share statistical patterns with the training examples (similar
vocabulary, similar structure) without needing an exact phrase match.

Deliberately simple model choice (TF-IDF + Logistic Regression, not a
transformer/deep learning model): interpretable (you can inspect which
words drive the decision via the model's coefficients), fast to train
and run with no GPU, and — importantly — a genuinely reasonable choice
for this problem size. Reaching for a large model here would be over-
engineering for a demo dataset of ~60 examples; matching model complexity
to problem/data size is itself the correct ML engineering judgment call.
"""
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "labeled_prompts.csv")
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "classifier.pkl")


def load_training_data(path: str = DEFAULT_DATA_PATH) -> tuple:
    df = pd.read_csv(path)
    return df["text"].tolist(), df["label"].tolist()


def train_classifier(texts: list, labels: list, test_size: float = 0.25, random_state: int = 42) -> dict:
    """
    Returns a dict with the trained vectorizer, model, and HONEST
    evaluation metrics on a held-out test split — not metrics on the
    training data itself, which would overstate real-world performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels,
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, pos_label="injection", zero_division=0),
        "recall": recall_score(y_test, predictions, pos_label="injection", zero_division=0),
        "f1": f1_score(y_test, predictions, pos_label="injection", zero_division=0),
        "test_set_size": len(y_test),
        "train_set_size": len(y_train),
    }

    return {"vectorizer": vectorizer, "model": model, "metrics": metrics}


def predict(text: str, vectorizer, model) -> dict:
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]
    class_index = list(model.classes_).index(prediction)
    confidence = probabilities[class_index]
    return {"label": prediction, "confidence": float(confidence)}


def save_model(trained: dict, path: str = DEFAULT_MODEL_PATH) -> None:
    with open(path, "wb") as f:
        pickle.dump({"vectorizer": trained["vectorizer"], "model": trained["model"]}, f)


def load_model(path: str = DEFAULT_MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def top_predictive_terms(trained: dict, n: int = 10) -> dict:
    """
    Returns the top N terms most associated with each class, by logistic
    regression coefficient magnitude — makes the model's decisions
    inspectable rather than a black box, which matters a lot for a
    security tool (you want to be able to explain WHY something was
    flagged, not just that it was).
    """
    vectorizer = trained["vectorizer"]
    model = trained["model"]
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]

    injection_class_is_positive = model.classes_[1] == "injection"
    sign = 1 if injection_class_is_positive else -1

    top_injection_indices = (sign * coefficients).argsort()[-n:][::-1]
    top_benign_indices = (sign * coefficients).argsort()[:n]

    return {
        "top_injection_terms": [feature_names[i] for i in top_injection_indices],
        "top_benign_terms": [feature_names[i] for i in top_benign_indices],
    }
