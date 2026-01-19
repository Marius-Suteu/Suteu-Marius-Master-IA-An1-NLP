from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_advanced(y_true, y_pred, y_probs=None):
    """
    y_true  : etichete reale
    y_pred  : etichete prezise
    y_probs : probabilitati pentru clasa pozitiva (FAKE)
    """

    print(classification_report(y_true, y_pred, digits=4))

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Macro-F1: {macro_f1:.4f}")

    if y_probs is not None:
        roc_auc = roc_auc_score(y_true, y_probs)
        pr_auc = average_precision_score(y_true, y_probs)
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC : {pr_auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
