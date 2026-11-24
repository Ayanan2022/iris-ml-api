"""Train a simple classifier on the Iris dataset and save it to model.joblib."""
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_PATH = "model.joblib"


def main() -> None:
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {accuracy:.3f}")

    # Save the model together with the class names so the API can return readable labels
    joblib.dump({"model": model, "target_names": list(iris.target_names)}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
