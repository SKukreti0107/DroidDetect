import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, balanced_accuracy_score, roc_auc_score,
                             ConfusionMatrixDisplay, RocCurveDisplay, roc_curve)
import matplotlib.pyplot as plt
import tabulate

# Load the dataset
data = pd.read_csv('TUANDROMD.csv')

# Drop any missing values
data = data.dropna()

# Split features and labels
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # Last column (assumed to be the label)

# Encode the labels if they are categorical
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define pipelines for each model
models = {
    'Logistic Regression': Pipeline([
        ('feature_selection', SelectKBest(chi2, k=10)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ]),
    'kNN': Pipeline([
        ('feature_selection', SelectKBest(chi2, k=10)),
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier())
    ]),
    'Decision Tree': Pipeline([
        ('feature_selection', SelectKBest(chi2, k=10)),
        ('classifier', DecisionTreeClassifier())
    ]),
    'SVM': Pipeline([
        ('feature_selection', SelectKBest(chi2, k=10)),
        ('scaler', StandardScaler()),
        ('classifier', SVC(probability=True))
    ]),
    'Naive Bayes': Pipeline([
        ('feature_selection', SelectKBest(chi2, k=10)),
        ('classifier', GaussianNB())
    ])
}

# Evaluate models using cross-validation and collect predictions
results = {}

for name, model in models.items():
    # Cross-validated predictions
    y_pred = cross_val_predict(model, X, y_encoded, cv=cv)
    
    # Cross-validated predicted probabilities
    if hasattr(model.named_steps['classifier'], "predict_proba"):
        y_proba = cross_val_predict(model, X, y_encoded, cv=cv, method='predict_proba')[:, 1]
    else:
        y_proba = cross_val_predict(model, X, y_encoded, cv=cv, method='decision_function')

    # Compute metrics
    accuracy = accuracy_score(y_encoded, y_pred)
    precision = precision_score(y_encoded, y_pred)
    recall = recall_score(y_encoded, y_pred)
    f1 = f1_score(y_encoded, y_pred)
    balanced_acc = balanced_accuracy_score(y_encoded, y_pred)
    roc_auc = roc_auc_score(y_encoded, y_proba)

    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc
    }

    # Plot confusion matrix
    cm = confusion_matrix(y_encoded, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {name}')
    plt.show()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_encoded, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'ROC Curve for {name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

# Print detailed results
def print_detailed_results(results):
    # Create a formatted table of results
    print("\n\n--- Cross-Validated Model Performance ---")
    table_data = []
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Balanced Accuracy', 'ROC AUC']
    
    for name, metrics in results.items():
        row = [
            name,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1_score']:.4f}",
            f"{metrics['balanced_accuracy']:.4f}",
            f"{metrics['roc_auc']:.4f}"
        ]
        table_data.append(row)
        
    # Print the table using tabulate
    print(tabulate.tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Identify and highlight the best model
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\n\n🏆 Best Performing Model: {best_model}")
    print("Detailed Performance of Best Model:")
    best_metrics = results[best_model]
    for metric, value in best_metrics.items():
        print(f"{metric}: {value:.4f}")
    
print_detailed_results(results)