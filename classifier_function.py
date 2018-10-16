import matplotlib.pyplot as plt
from sklearn import tree, ensemble, linear_model
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc,precision_recall_curve, confusion_matrix

def feature (df):
    y_ = df.profitable.values
    x_ = df.drop('profitable', axis = 1).values
    x_names = df.drop('profitable', axis = 1).columns

    return x_, x_names, y_


def run_classifier (classifier, x_, y_, cv):
    probas_cv = []
    y_pred_cv = []

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fpr_cv = []
    tpr_cv = []
    thresholds_cv = []

    precision_cv = []
    recall_cv = []
    thresholds_pr_cv = []

    y_test_cv = []

    for train, test in cv.split(x_, y_):

        y_test_cv.append(y_[test])

        probas_ = classifier.fit(x_[train], y_[train]).predict_proba(x_[test])
        probas_cv.append(probas_)

        y_pred = classifier.fit(x_[train], y_[train]).predict(x_[test])
        y_pred_cv.append(y_pred)

        fpr, tpr, thresholds = roc_curve(y_[test], probas_[:, 1])
        fpr_cv.append(fpr)
        tpr_cv.append(tpr)
        thresholds_cv.append(thresholds)

        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        precision, recall, threshold = precision_recall_curve(y_[test], probas_[:, 1])
        precision_cv.append(precision)
        recall_cv.append(recall)
        thresholds_pr_cv.append(threshold)

    return probas_cv, y_test_cv, y_pred_cv,tprs, aucs, fpr_cv, tpr_cv, thresholds_cv, precision_cv, recall_cv


