import argparse
import os
import sklearn as sk
import logging as log
import numpy as np
from pathlib import Path

import classifier as cl
import download_GDC_data as dgdc
import preprocess as pr
import unsupervised_classifier as uc
import genetic as g

Path.ls = lambda x: [o.name for o in x.iterdir()]

def str2bool(value):
    return value.lower() == 'true' or value == '1'


def init_logger():
    log.basicConfig(format='%(asctime)s - %(message)s', level=log.INFO)
    logFormatter = log.Formatter("%(asctime)s [%(levelname)s]  %(message)s")
    rootLogger = log.getLogger()

    log_file = "./log"
    fileHandler = log.FileHandler("{0}/{1}.log".format("./", log_file))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # consoleHandler = log.StreamHandler()
    # consoleHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(consoleHandler)
    return log_file + ".log"


log_file = init_logger()
print = log.info


def main():
    p = Path("./result")
    if not p.exists():
        os.makedirs(p)

    parser = argparse.ArgumentParser(description='Bioinf project. The arguments can be passed in any order.')

    classes = parser.add_mutually_exclusive_group()
    classes.add_argument('-cl2',
                         help='in order to classify two cancer types.',
                         action='store_true')
    classes.add_argument('-cl3',
                         help='in order to classify two cancer types AND sane.',
                         action='store_true')
    
    classifier = parser.add_mutually_exclusive_group()
    classifier.add_argument('-svm', help='train a Support Vector Machine classifier', action='store_true')
    classifier.add_argument('-knn', help='train a K Nearest Neighbors classifier', action='store_true')
    classifier.add_argument('-rforest', help='train a Random Forest classifier', action='store_true')
    classifier.add_argument('-kmeans', help='train a Kmeans clustering', action='store_true')
    classifier.add_argument('-hierarc', help='train an Agglomerative Hierarchical clustering', action='store_true')

    inbalance = parser.add_mutually_exclusive_group()
    inbalance.add_argument('-over', help='imbalance: Random Oversampling ', action='store_true')
    inbalance.add_argument('-smote', help='imbalance: SMOTE', action='store_true')

    preprocess = parser.add_mutually_exclusive_group()
    preprocess.add_argument('-ttest',
                            help='feature selection: ttest per chromosoma and per cpg site - 2 classes',
                            action='store_true')
    preprocess.add_argument('-fisher',
                            help='feature selection: fisher criterion - 3 classes',
                            action='store_true')
    preprocess.add_argument('-anova',
                            help='feature selection: anova - 3 classes',
                            action='store_true')    
    preprocess.add_argument('-pca',
                            help='dimensionality reduction: Principal Component Analisys',
                            action='store_true')
    preprocess.add_argument('-lda',
                            help='dimensionality reduction: Linear Discriminant Analysis',
                            action='store_true')
    preprocess.add_argument('-sfs',
                            help='feature selection - wrapper: Step Forward Selection (nearly unfeasible)',
                            action='store_true')
    preprocess.add_argument('-ga', help='feature selection - wrapper: Genetic Algorithm', 
                            action='store_true')

    parser.add_argument('-d', '--download', nargs=2,
                        help='download Adenoma and Adenocarcinoma and Squamous Cell Neoplasm '+
                            'data from Genomic Data Common. It needs 2 parameters: '+
                        'first parameter is the destination folder; '+
                         'second parameters is the number of files to be downloaded for each class ',
                        action='store')
    parser.add_argument('-ds', '--downloadsane', nargs=2,
                        help='download Sane data from Genomic Data Common'+
                        'It needs 2 parameters: '+
                        'first parameter is the destination folder; '+
                         'second parameters is the number of files to be downloaded ',
                        action='store')
    parser.add_argument('-s', '--store',
                        help='concatenate files belonging to same cancer type and store them in a binary file',
                        action='store')


    parser.add_argument('--alpha', type=float, default=0.001,
                        help='to set a different ALPHA: ttest parameter - default is 0.001',
                        action='store')
    parser.add_argument('--perc', type=float, default=0.95,
                        help='to set PERC of varaince explained by the features kept by PCA',
                        action='store')
    parser.add_argument('-rs','--r_state', type=int, default=8,
                        help='to set a user defined Random State - default is 8',
                        action='store')
    parser.add_argument('--only_chrms_t', default=False,
                        help='select only chrms for ttest',
                        action='store_true')
    parser.add_argument('--crossval',
                        help='to do crossvalidation OR in case of unsupervised to plot the Inertia curve',
                        action='store_true')
    parser.add_argument('--plot_lc',
                        help='plot the learning curve',
                        action='store_true')
    parser.add_argument('--remove_nan_cpgs', type=str2bool, default=True,
                        help='IF True: removes features containing at least one NaN value. '+
                            'IF False: NaN are substituted by the mean over the feature. '+
                            'The old file resulted by feature reduction must be eliminated when changing option. '+
                             'By Default is True.',
                        action='store')

    args = parser.parse_args()

    if args.download:
        print("download ")
        dgdc.getDataEx(path=args.download[0], file_n=args.download[1])
    if args.downloadsane:
        print("download sane ")
        dgdc.getSaneDataEx(path=args.downloadsane[0], file_n=args.downloadsane[1])
    if args.store:
        print("store")
        dgdc.storeDataIntoBinary(path=args.store)
        print("Data stored.")

    # validity checks
    if not args.cl2 and not args.cl3:
        print("insert arg -cl2 for classifying 2 classes OR -cl3 for 3 classes")
        return

    # parameters and variables
    alpha = args.alpha  # alpha parameter for t-test
    perc = args.perc  # percentage of variance explained
    classes = 2 if args.cl2 else 3
    random_state = args.r_state
    no_nan = args.remove_nan_cpgs
    n_components=100

    cl.setPlot_lc(args.plot_lc)

    cl.addToName("cl{}".format(classes))
    cl.addToName("rs{}".format(random_state))

    # load data
    print("Loading....")
    x, y, chrms_pos = pr.loadData(classes=classes)
    if no_nan:
        cl.addToName("no_nan")
        length = x.shape[1]
        x = pr.removeNanFeature(x)
        print("{} NaN features removed!".format(length - x.shape[1]))
    print("Loaded!")

    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.2,
                                                                           random_state=random_state)
    del x, y

    # preprocess
    if args.ttest:
        if classes != 2:
            print("wrong number of classes")
            return
        #print("Start ttest axis={}....".format(args.ttest))
        r, cpg_r = pr.compute_t_test(x_train, y_train, chrms_pos, alpha, random_state, axis=0,
                                     remove_nan=no_nan)
        print(r)
        cl.addToName("ttest{}".format(args.ttest))
        length = x_train.shape[1]
        x_train, x_test = pr.removeFeatures(x_train, x_test, cpg_r, chrms_pos, args.only_chrms_t,
                                            remove_nan=no_nan, y_train=y_train)
        print("Features removed: {}".format(length - x_train.shape[1]))
        print("End ttest!")
        
    if args.ga:
        print("genetic algorithm")
        cl.addToName("ga")
        # per lavorare con meno componenti
        # x_train = x_train[:, 1:100]
        result = g.GA_function(x_train, y_train, random_state, classes, 0.1)
        path = Path('./data/GA_{}_{}.npy'.format(random_state, classes))
        np.save(path, result)
        x_train = x_train[:, result]
        x_test = x_test[:, result]

    if args.pca:
        print("pca")
        cl.addToName("pca")
        x_train, x_test = pr.pca_function(x_train, x_test, y_train, y_test, classes, perc, random_state, name=cl.name,
                                          remove_nan=no_nan)

    if args.lda:
        #print("lda - {} components".format(args.lda))
        cl.addToName("lda")
        x_train, x_test = pr.lda_function(x_train, x_test, y_train, y_test, classes, args.lda, random_state, cl.name)

    if args.fisher:
        if classes != 2:
            print("wrong number of classes")
            return
        #cl.addToName("fisher{}".format(args.fisher))
        cl.addToName("fisher")
        print("fisher")
        x_train, x_test = pr.fisher_function(x_train, x_test, y_train, y_test, random_state, best=True,n=n_components,
                                             remove_nan=no_nan)
        # if best=True selects the n best features, if False the worst n features (for debugging)
    if args.sfs:
        if classes != 2:
            print("wrong number of classes")
            return
        print("Start sfs....")
        feat_col = pr.sfs(x_train, x_test, y_train, y_test, chrms_pos, alpha, random_state)
        x_train = x_train[:, feat_col]
        x_test = x_test[:, feat_col]

    if args.anova:
        if classes != 3:
            print("wrong number of classes")
            return
        print("anova")
        cl.addToName("anova")
        x_train, x_test = pr.anova_function(x_train, x_test, y_train, y_test, alpha, random_state, remove_nan=no_nan)

    # imbalance
    if args.over:
        print("over ")
        x_train, y_train = pr.imbalance(x_train, y_train, "over", random_state)
        cl.addToName("over")

    if args.smote:
        print("smote ")
        x_train, y_train = pr.imbalance(x_train, y_train, "smote", random_state)
        cl.addToName("smote")

    cl.random_state(random_state)

    # classify
    if args.svm:
        print("svm ")
        cl.svm(x_train, x_test, y_train, y_test, classes=classes, crossval=args.crossval)

    if args.knn:
        print("knn ")
        cl.knn(x_train, x_test, y_train, y_test, classes=classes, crossval=args.crossval)

    if args.rforest:
        print("rforest")
        cl.random_forest(x_train, x_test, y_train, y_test, classes=classes, crossval=args.crossval)

    if args.kmeans:
        print("kmeans")
        uc.kmeans(x_train, x_test, y_train, y_test, classes=classes, random_state=random_state, crossval=args.crossval)

    if args.hierarc:
        print("hierarchical clustering")
        uc.hierarchical(x_train, x_test, y_train, y_test, classes=classes, random_state=random_state,
                        crossval=args.crossval)

    print("Log name: {}.log".format(cl.name))

    handlers = log.getLogger().handlers[:]
    for handler in handlers:
        handler.close()
        log.getLogger().removeHandler(handler)
    nf = p / cl.name
    if not nf.exists():
        os.makedirs(nf)
    npath = Path(nf / '{}.log'.format(cl.name))
    i = 1
    while npath.exists():
        npath = Path(nf / '{}_{}.log'.format(cl.name, i))
        i += 1
    os.rename('log.log', npath)


if __name__ == '__main__':
    main()
