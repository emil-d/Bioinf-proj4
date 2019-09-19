'''
module to perform data classification
'''
import logging as log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import os
import plot

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from pathlib import Path
from scipy import interp
from itertools import cycle

print = log.info
name = ""
plot_lc = False
random = 10

def setPlot_lc(plot_v):
    global plot_lc
    plot_lc = plot_v


def random_state(rs):
    global random
    random = rs


def addToName(str):
    global name
    name = "{}_{}".format(name, str)


def save_roc_val(fpr, tpr, file):
    with h5py.File(file, 'w') as hf:
        hf.create_dataset("fpr", data=fpr)
        hf.create_dataset("tpr", data=tpr)


def custom_cv_2folds(x_train, x_test):
    n = [0, x_train, x_test + x_train]
    i = 1
    while i <= 2:
        idx = np.arange(n[i - 1], n[i], dtype=int)
        yield idx, idx
        i += 1


def gscv(model, params, x_train, x_test, y_train, y_test, cv=5, classes=2, crossval=True):
    global name
    print("Start_Classes{}".format(classes))
    name = "{}{}".format(type(model).__name__, name)
    p = Path("./result") / name
    if not p.exists():
        os.makedirs(p)
    global plot_lc
    global random
    if plot_lc:
        exploit_incremental_learning = False
        pl = plot.plot_learning_curve(model, name, np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)),
                                      (0, 1.01), custom_cv_2folds(x_train.shape[0], x_test.shape[0]), n_jobs=1,
                                      random_state=random, exploit_incremental_learning=exploit_incremental_learning)
        pl.savefig(p / "{}_learning_curve".format(name))
    elif crossval:
        gscv = GridSearchCV(model, params, cv=cv, verbose=2, return_train_score=True, n_jobs=2)

        gscv.fit(x_train, y_train)
        print("Best params: {}".format(gscv.best_params_))
        print("Best score: {}".format(gscv.best_score_))
        res = pd.DataFrame.from_dict(gscv.cv_results_)
        print("Result: {}".format(res))
        res.to_csv(p / "{}.csv".format(name))
        res.to_html(p / "{}.html".format(name))
        print("Accuracy on test:{}".format(gscv.score(x_test, y_test)))

        if classes == 2:
            if type(model).__name__ == 'RandomForestRegressor':
                probs = gscv.predict(x_test)
                print("F1 on test:{}".format(metrics.f1_score(y_test, np.round(probs))))
            else:
                probs = gscv.predict_proba(x_test)[:, 1]
                print("F1 on test:{}".format(metrics.f1_score(y_test, gscv.predict(x_test))))
            print("AUC on test:{}".format(metrics.roc_auc_score(y_test, probs)))
            fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
            save_roc_val(fpr, tpr, p / "{}_ROC.h5".format(name))
            plt.plot(fpr, tpr)
            plt.title('ROC {}'.format(name))
            plt.ylabel('TPR')
            plt.xlabel('FPR')
            plt.savefig(p / "{}_ROC".format(name))
        else:
            probs = gscv.predict(x_test) #predict_proba(x_test)[:, 1]
            print("F1 on test (macro):{}".format(metrics.f1_score(y_test, probs, average='macro')))

    else:
        model.fit(x_train, y_train)
        print("Accuracy on test:{}".format(model.score(x_test, y_test)))
        if classes == 2:
            probs = model.predict(x_test)  # predict_proba(x_test)[:, 1]
            print("F1 on test:{}".format(metrics.f1_score(y_test, model.predict(x_test))))
            print("AUC on test:{}".format(metrics.roc_auc_score(y_test, probs)))
            fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)
            save_roc_val(fpr, tpr, p / "{}_ROC.h5".format(name))
            plt.plot(fpr, tpr)
            plt.title('ROC {}'.format(name))
            plt.ylabel('TPR')
            plt.xlabel('FPR')
            plt.savefig(p / "{}_ROC".format(name))
        else:
            probs = model.predict(x_test)  # .predict_proba(x_test)[:, 1]#gscv.
            print("F1 on test (macro):{}".format(metrics.f1_score(y_test, probs, average='macro')))

            probs = model.predict_proba(x_test)

            # Compute ROC curve and ROC area for each class        
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(classes):
                fpr[i], tpr[i], _ = metrics.roc_curve(y_test, probs[:, i], pos_label=i)
                roc_auc[i] = metrics.auc(fpr[i], tpr[i])

            # Plot of a ROC curve for specific classes
            for i in range(classes):
                plt.figure()
                plt.plot(fpr[i], tpr[i], color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc[i])
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC {}'.format(name))
                plt.legend(loc="lower right")
                plt.savefig(p / "{}_ROC_{}".format(name, i), dpi=1200)

            # Compute macro-average ROC curve and ROC area

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
            print("AUC on test:{}".format(roc_auc["macro"]))

            # Plot all ROC curves
            plt.figure()
            
            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=1)

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=1,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Some extension of Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")
            plt.savefig(p / "{}_ROC_aggregate".format(name), dpi=1200)


def knn(x_train, x_test, y_train, y_test, classes=2, crossval=True, n_neighbor=[1, 3, 6, 9]):
    knn = KNeighborsClassifier()
    if classes == 2:
        best_k = 1
    else:
        best_k = 3
    if crossval:
        params = {"n_neighbors": n_neighbor}
    else:
        params = {"n_neighbors": best_k}
        knn = KNeighborsClassifier(**params)
    gscv(knn, params, x_train, x_test, y_train, y_test, classes=classes, crossval=crossval)


def svm(x_train, x_test, y_train, y_test, classes=2, crossval=True, c=np.array([0.1, 1, 10]),
        kernel=['linear']):  # 'poly', 'rbf', 'sigmoid'
    # SVM
    bestC = 0.1
    svm = SVC(probability=True, gamma='scale')
    if crossval:
        params = {"C": c, "kernel": kernel}
    else:
        params = {"C": bestC, "kernel": 'linear'}
        svm = SVC(probability=True, **params) 
    gscv(svm, params, x_train, x_test, y_train, y_test, classes=classes, crossval=crossval)


def random_forest(x_train, x_test, y_train, y_test, classes=2, crossval=True, n_estimators=range(50, 250, 50),
                  max_depth=range(5, 25, 5), leaves=range(50, 125, 25)):
    # y_test=np.random.randint(2, size=y_test.shape[0])
    # y_train=np.random.randint(2, size=y_train.shape[0])
    model = RandomForestClassifier()  # RandomForestRegressor()  # random_state= seed for random function
    if crossval:
        params = {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_leaf": leaves}
    else:
        if classes == 2:
            params = {"n_estimators": 50, "max_depth": 5, "min_samples_leaf": 50}  # fit with the best parameter combo
        else:
            params = {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 50}  # fit with the best parameter combo
        model = RandomForestClassifier(**params)
    gscv(model, params, x_train, x_test, y_train, y_test, classes=classes, crossval=crossval)
