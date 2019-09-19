'''
module to perform data preprocessing
'''
import json
import logging as log
import pickle
import zlib
from pathlib import Path

import plot as p
import classifier as cl

import h5py
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.decomposition import PCA
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

print = log.info

Path.ls = lambda x: [o.name for o in x.iterdir()]


def loadData(path="./data", classes=2):
    path_chrms = Path(path) / "chrms.dat"
    path_h5 = Path(path) / "data.h5"
    if not path_chrms.is_file() or not path_h5.is_file():
        print("FILE ERROR")
        exit(0)
    with open(path_chrms) as json_file:
        chrms = json.load(json_file)
    with h5py.File(path_h5, 'r') as hf:
        X = None
        Y = np.zeros(0)
        for idx, key in enumerate(["Adenomas and Adenocarcinomas", "Squamous Cell Neoplasms", "sane"][:classes]):
            tmp = hf[key]
            Y = np.concatenate((Y, np.ones(len(tmp)) * idx))
            if X is None:
                X = tmp
            else:
                X = np.concatenate((X, tmp))
    return X, Y, chrms


def removeFeatures(x_train, x_test, cpg_r, chrms_pos, only_chrms_t, remove_nan, y_train):
    features = None
    if remove_nan:
        path_h5 = Path("data/cpgs_nan.h5")
        if not path_h5.is_file():
            print("FILE ERROR")
            exit(0)
        with h5py.File(path_h5, 'r') as hf:
            features = np.array(hf["cpgs_nan"])
   
    with tqdm(total=len(cpg_r.keys())) as pbar:
        d = np.ones(0)
        for chrm in chrms_pos.keys():
            start = chrms_pos[chrm][0]
            end = chrms_pos[chrm][1]
            length = end - start
            if remove_nan:
                length -= len(features[start: end][features[start: end]])
                # end = start + length
            if len(cpg_r[chrm]) == 0:
                d = np.concatenate((d, np.zeros(length)))
            else:
                if only_chrms_t:
                    #### fix
                    d = np.concatenate((d,np.ones(length)))
                else:
                    a = np.asarray(cpg_r[chrm])
                    d = np.concatenate((d, a))
            pbar.update()
        
        if x_train[:, d == 1].shape[1] != 0 and x_train[:, d == 0].shape != 0:
            p.plot_histogram(x_train[:, d == 1][0], y_train,
                             "T-test Significative Feature Distribution", "{}_distribution_signif".format(cl.name), 2)
            p.plot_histogram(x_train[:, d == 0][0], y_train,
                             "T-test Non-significative Feature Distribution", "{}_distribution_nonsign".format(cl.name),
                             2)

        x_train = x_train[:, d == 1]
        x_test = x_test[:, d == 1]
    return x_train, x_test


def imbalance(X, Y, type, rstate):
    if type == 'smote':
        tl = SMOTE(random_state=rstate)
        
    elif type == 'over':
        tl = RandomOverSampler(random_state=rstate)

    X_return, Y_return = tl.fit_sample(X, Y)
    return X_return, Y_return


def t_test_function(A, B, alpha, mean=True, axis=0):
    if mean:
        A = np.mean(A, axis=axis)
        B = np.mean(B, axis=axis)

    t_value, p_value = stats.ttest_ind(A, B, equal_var=False)
    if p_value > alpha:  # non esprime differrenziazione tra i due tumori
        result = 0
    else:
        result = 1  # esprime differenziazione tra i due tumori
    return result


def compute_t_test(x_train, y_train, chrms_pos, alpha= 0.001, random_state=8, axis=0, remove_nan=True):
    path = Path(
        './data/t_test_res_axis{}_{}_{}{}.dat'.format(axis, random_state, alpha, "_no_nan" if remove_nan else ""))
    if path.is_file():
        compressed_data = open(path, 'rb').read()
        decompressed_data = zlib.decompress(compressed_data)
        r, cpg_r = pickle.loads(decompressed_data)
        return r, cpg_r

    r = {}
    cpg_r = {}
    if remove_nan:
        path_h5 = Path("data/cpgs_nan.h5")
        if not path_h5.is_file():
            print("FILE ERROR")
            exit(0)
        with h5py.File(path_h5, 'r') as hf:
            features = np.array(hf["cpgs_nan"])
    tr = 0
    with tqdm(total=len(chrms_pos) + x_train.shape[1]) as pbar:
        for chrm in chrms_pos.keys():
            start = chrms_pos[chrm][0]
            end = chrms_pos[chrm][1]
            length = end - start
            if remove_nan:
                removed = len(features[start: end][features[start: end]])
                length -= removed
                start -= tr
                tr += removed
                end = start + length
            
            if chrm=='chr18' or chrm =='chr21':
                import pdb; pdb.set_trace()
                
            r[chrm] = t_test_function(x_train[y_train == 0, start:end],
                                      x_train[y_train == 1, start:end], alpha, axis=axis)
            cpg_r[chrm] = []
            if r[chrm] == 1:
                for i_cpg in range(length):
                    cpg_r[chrm].append(t_test_function(x_train[y_train == 0, start + i_cpg],
                                                       x_train[y_train == 1, start + i_cpg], alpha,
                                                       mean=False))
                    pbar.update()
            else:
                pbar.update(length)
            pbar.update()

    compressed = zlib.compress(pickle.dumps((r, cpg_r)), zlib.Z_BEST_COMPRESSION)
    f = open(path, 'wb')
    f.write(compressed)
    f.close()
    return r, cpg_r


def anova_function(x_train, x_test, y_train, y_test, alpha, r_state, remove_nan):
    path = Path('./data/anova_{}_{}{}.npy'.format(r_state, alpha, "_no_nan" if remove_nan else ""))
    if path.is_file():
        result = np.load(path)
    else:
        result = []

        dfs_ad = x_train[y_train == 0, :]
        dfs_sq = x_train[y_train == 1, :]
        dfs_sane = x_train[y_train == 2, :]

        for column in range(x_train.shape[1]):
            F_statistic, p_value = stats.f_oneway(dfs_ad[:, column], dfs_sq[:, column], dfs_sane[:, column])
            if p_value >= alpha:  # il cg non esprime differrnziazione tra i due tumori
                result.append(0)
            else:
                result.append(1)  # il cg esprime differenziazione tra i due tumori
        result = np.asarray(result)
        np.save(path, result)

        if x_train[:, result == 1].shape[1] != 0 and x_train[:, result == 0].shape != 0:
            p.plot_histogram(x_train[:, result == 1][0], y_train,
                             "Anova Significative Feature Distribution", "{}_distribution_signif".format(cl.name), 3)
            p.plot_histogram(x_train[:, result == 0][0], y_train,
                             "Anova Non-significative Feature Distribution", "{}_distribution_nonsign".format(cl.name),
                             3)

    x_train = x_train[:, result == 1]
    x_test = x_test[:, result == 1]

    return x_train, x_test

def fisher_function(x_train, x_test, y_train, y_test, r_state,best,n,remove_nan):
    n = int(n)
    path = Path('./data/fisher_{}{}.npy'.format(r_state, "_no_nan" if remove_nan else ""))
    if path.is_file():
        result = np.load(path)
    else:
        result = []
        #numero_sq = int(sum(y_train))
        #numero_ad = y_train.shape[0] - numero_sq

        dfs_ad = x_train[y_train == 0, :]
        dfs_sq = x_train[y_train == 1, :]

        for column in range(x_train.shape[1]):
            value = (np.mean(dfs_ad[:, column]) - np.mean(dfs_sq[:, column])) ** 2 / (
                    np.var(dfs_ad[:, column]) + np.var(dfs_sq[:, column]))
            result.append(value)
        best_feat = result.index(max(result))
        worste_feat = result.index(min(result))
        print(best_feat)
        print(worste_feat)

        result = np.asarray(result)
        np.save(path, result)

        p.plot_histogram(x_train[:, best_feat], y_train,
                         "Fisher Best Feature Distribution", "{}_distribution_best".format(cl.name), 2)
        p.plot_histogram(x_train[:, worste_feat], y_train,
                         "Fisher Worst Feature Distribution", "{}_distribution_worst".format(cl.name), 2)

    if best:
        result_max = result[np.argsort(result)[-n:]]
        indice = np.where(result >= min(result_max))[0]
    else:
        result_max = result[np.argsort(result)[:n]]
        indice = np.where(result <= max(result_max))[0]

    x_train = x_train[:, indice]
    x_test = x_test[:, indice]

    return x_train, x_test


def pca_transform(ds, mean, T):
    if mean is not None:
        ds = ds - mean
    return np.dot(ds, T)


def pca_function(train, test, label_train, label_test, classe, perc, r_state, remove_nan, name="default"):
    pathT = Path('./data/pca_cl{}_{}_{}{}_T.npy'.format(classe, r_state, perc, "_no_nan" if remove_nan else ""))
    pathM = Path('./data/pca_cl{}_{}_{}{}_M.npy'.format(classe, r_state, perc, "_no_nan" if remove_nan else ""))
    if pathT.is_file() and pathM.is_file():
        mean = np.load(pathM)
        T = np.load(pathT)
        return pca_transform(train, mean, T), pca_transform(test, mean, T)
    pca = PCA(perc, random_state=r_state)
    pca.fit(train)
    train = pca.transform(train)
    if len(test) != 0:
        test = pca.transform(test)
    plt.plot(np.array([pca.explained_variance_ratio_[:i].sum() for i in range(0, pca.components_.shape[0])]))
    plt.title('Variance explained')
    plt.ylabel('cumulatve variance explained')
    plt.xlabel('number of PC useds')
    plt.savefig("./result/{}_curva_Varianze_explained".format(name))

    print("PCA ratio: {}".format(pca.explained_variance_ratio_))

    np.save(pathT, pca.components_.T)
    np.save(pathM, pca.mean_)

    return train, test


def lda_function(train, test, label_train, label_test, classe, n_comp, r_state, name="default"):
    # lda = LDA(n_components=int(n_comp))
    lda = LDA()
    lda.fit(train, label_train)

    train = lda.transform(train)
    test = lda.transform(test)

    plt.plot(np.array([lda.explained_variance_ratio_[:i].sum() for i in range(0, int(n_comp))]))
    plt.title('Variance explained')
    plt.ylabel('cumulatve variance explained')
    plt.xlabel('number of Components useds')
    plt.savefig("./result/{}_curva_Varianze_explained".format(name))

    return train, test


def removeNanFeature(x, path="./data"):
    path_h5 = Path(path) / "cpgs_nan.h5"
    if not path_h5.is_file():
        print("FILE ERROR")
        exit(0)
    with h5py.File(path_h5, 'r') as hf:
        feature = np.array(hf["cpgs_nan"])
        x = x[:, np.logical_not(feature)]
    return x
