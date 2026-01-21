
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


RANDOM_SEED = 42


def load_and_explore_wine_data():
    """
    Load the wine dataset and perform initial exploration
    Returns features (X), target (y), and the wine dataset object
    """
    # Load the wine dataset from scikit-learn
    wine = load_wine()
    X = wine.data
    y = wine.target

    # Create a pandas DataFrame for easier manipulation
    df = pd.DataFrame(data=np.c_[wine['data'], wine['target']],
                      columns=wine['feature_names'] + ['target'])

    # Display dataset information
    print("=" * 60)
    print("WINE CULTIVAR DATASET EXPLORATION")
    print("=" * 60)
    print(f"Number of samples: {df.shape[0]}")
    print(f"Number of features: {df.shape[1] - 1}")
    print(f"Number of classes (cultivars): {len(wine.target_names)}")
    print("\nClass Names:", wine.target_names)
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nStatistical Summary:")
    print(df.describe())

    return X, y, wine, df


def preprocess_data(X, y):
    """
    Split the data into training and testing sets and scale features
    Returns processed training and testing data
    """
    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Scale the features for models that require it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def train_and_evaluate_logistic_regression(X_train, X_test, y_train, y_test, wine):
    """
    Train and evaluate a Logistic Regression model
    Returns the trained model and its predictions
    """
    print("\n" + "=" * 60)
    print("MODEL 1: LOGISTIC REGRESSION")
    print("=" * 60)

    # Initialize and train the Logistic Regression model
    log_reg = LogisticRegression(max_iter=200, random_state=RANDOM_SEED)
    log_reg.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = log_reg.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nLogistic Regression Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))

    return log_reg, y_pred, accuracy


def train_and_evaluate_decision_tree(X_train, X_test, y_train, y_test, wine):
    """
    Train and evaluate a Decision Tree model
    Returns the trained model and its predictions
    """
    print("\n" + "=" * 60)
    print("MODEL 2: DECISION TREE CLASSIFIER")
    print("=" * 60)

    # Initialize and train the Decision Tree model
    dt_classifier = DecisionTreeClassifier(random_state=RANDOM_SEED)
    dt_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = dt_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nDecision Tree Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))

    return dt_classifier, y_pred, accuracy


def train_and_evaluate_svm(X_train, X_test, y_train, y_test, wine):
    """
    Train and evaluate a Support Vector Machine model
    Returns the trained model and its predictions
    """
    print("\n" + "=" * 60)
    print("MODEL 3: SUPPORT VECTOR MACHINE")
    print("=" * 60)

    # Initialize and train the SVM model
    svm_classifier = SVC(random_state=RANDOM_SEED)
    svm_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nSVM Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=wine.target_names))

    return svm_classifier, y_pred, accuracy


def visualize_results(best_model, best_model_name, X_test, y_test, wine, df):
    """
    Visualize the results using confusion matrix and feature importance
    """
    print("\n" + "=" * 60)
    print("VISUALIZING MODEL RESULTS")
    print("=" * 60)

    # Make predictions with the best model
    y_pred = best_model.predict(X_test)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names,
                yticklabels=wine.target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {best_model_name}')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Feature importance (only for models that support it)
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = pd.Series(
            best_model.feature_importances_,
            index=wine.feature_names
        )
        feature_importances = feature_importances.sort_values(ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x=feature_importances, y=feature_importances.index)
        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title(f"Important Features for Wine Cultivar Prediction ({best_model_name})")
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()


def compare_models(accuracies, model_names):
    """
    Compare the performance of all models
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies
    })

    # Sort by accuracy
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

    # Display the comparison
    print(comparison_df)

    # Visualize the comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=comparison_df)
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

    # Find the best model
    best_idx = np.argmax(accuracies)
    best_model_name = model_names[best_idx]
    print(f"\nBest performing model: {best_model_name} with accuracy {accuracies[best_idx]:.4f}")

    return best_model_name


def main():
    """
    Main function to run the wine cultivar prediction pipeline
    """
    # Step 1: Load and explore the data
    X, y, wine, df = load_and_explore_wine_data()

    # Step 2: Preprocess the data
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(X, y)

    # Step 3: Train and evaluate models
    # Note: We use scaled data for LR and SVM, but original data for Decision Tree
    log_reg, y_pred_lr, acc_lr = train_and_evaluate_logistic_regression(
        X_train_scaled, X_test_scaled, y_train, y_test, wine
    )

    dt_classifier, y_pred_dt, acc_dt = train_and_evaluate_decision_tree(
        X_train, X_test, y_train, y_test, wine
    )

    svm_classifier, y_pred_svm, acc_svm = train_and_evaluate_svm(
        X_train_scaled, X_test_scaled, y_train, y_test, wine
    )

    # Step 4: Compare models
    model_names = ['Logistic Regression', 'Decision Tree', 'SVM']
    accuracies = [acc_lr, acc_dt, acc_svm]
    best_model_name = compare_models(accuracies, model_names)

    # Step 5: Visualize results for the best model
    if best_model_name == 'Logistic Regression':
        best_model = log_reg
        X_test_vis = X_test_scaled
    elif best_model_name == 'Decision Tree':
        best_model = dt_classifier
        X_test_vis = X_test
    else:  # SVM
        best_model = svm_classifier
        X_test_vis = X_test_scaled

    visualize_results(best_model, best_model_name, X_test_vis, y_test, wine, df)


if __name__ == "__main__":
    main()