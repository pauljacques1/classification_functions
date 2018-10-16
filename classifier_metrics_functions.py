# This file contains all the functions for plotting the metrics of the classifier

import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
import itertools


def plot_roc_auc (cv, fpr_cv, tpr_cv, aucs, tprs):
    mean_fpr = np.linspace(0, 1, 100)
    for i in range (0,cv.n_splits):

        plt.plot(fpr_cv[i], tpr_cv[i], lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, aucs[i]))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def plot_pr_curve(cv, precision_cv, recall_cv):

    for i in range(0, cv.n_splits):
        plt.plot(recall_cv[i], precision_cv[i], lw=1, alpha=0.3, label='PR fold %d ' % (i))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall (TPR)')
    plt.ylabel('Precision')
    plt.title('Precision - Recall Curve')
    plt.legend(loc="lower right")
    plt.show()



def plot_feature_importance(classifier, x_, x_names, y_):
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_names[i] for i in indices]

    plt.bar(range(x_.shape[1]), importances[indices])
    plt.xticks(range(x_.shape[1]), names, rotation=90)
    plt.show()


classes = np.asarray([0,1])
def plot_confusion_matrix(classes, y_test_cv, y_pred_cv, x=0):
    cnf_matrix = confusion_matrix(y_test_cv[x], y_pred_cv[x])
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

