'''
module to perform plot
'''
import numpy as np
import matplotlib.pyplot as plt





def plot_histogram(feature, y, title, fname, classes):
    plt.figure()
    plt.title(title)
    plt.ylabel('Frequency')
    plt.xlabel('Beta Value')
    if classes == 2:
        plt.hist(feature[np.where(y == 0)], 50, facecolor='green', alpha=0.75)
        plt.hist(feature[np.where(y == 1)], 50, facecolor='red', alpha=0.75)
    else:
        plt.hist(feature[np.where(y == 0)], 50, facecolor='green', alpha=0.75)
        plt.hist(feature[np.where(y == 1)], 50, facecolor='red', alpha=0.75)
        plt.hist(feature[np.where(y == 2)], 50, facecolor='blue', alpha=0.75)
    plt.savefig("./result/{}".format(fname), dpi=1200)


from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), random_state=10,
                        exploit_incremental_learning=False):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, random_state=random_state,
        exploit_incremental_learning=exploit_incremental_learning)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    return plt
