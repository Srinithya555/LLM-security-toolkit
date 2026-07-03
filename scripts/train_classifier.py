"""
Run: python scripts/train_classifier.py
Trains the classifier on fixtures/labeled_prompts.csv, prints honest
held-out evaluation metrics, and saves the trained model for reuse.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_security.ml_classifier import load_training_data, train_classifier, save_model, top_predictive_terms


def main():
    texts, labels = load_training_data()
    print(f"Loaded {len(texts)} labeled examples "
          f"({labels.count('benign')} benign, {labels.count('injection')} injection)")

    trained = train_classifier(texts, labels)

    print("\n=== Held-out test set metrics ===")
    for key, value in trained["metrics"].items():
        print(f"  {key}: {value}")

    print("\n=== Most predictive terms (interpretability) ===")
    terms = top_predictive_terms(trained, n=10)
    print(f"  Injection-associated: {terms['top_injection_terms']}")
    print(f"  Benign-associated: {terms['top_benign_terms']}")

    save_model(trained)
    print(f"\nModel saved to fixtures/classifier.pkl")


if __name__ == "__main__":
    main()
