import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import f
from scipy.stats import t
from scipy.stats import spearmanr, rankdata
from scipy import stats
from IPython.display import display

## Матрица Пирсона
def corr_matrix_pirson(df):
    data = df.values
    n = data.shape[1]
    cols = df.columns
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x = data[:, i]
            y = data[:, j]
            xm = x.mean()
            ym = y.mean()
            num = ((x - xm) * (y - ym)).sum()
            den = np.sqrt(((x - xm)**2).sum() * ((y - ym)**2).sum())
            res[i, j] = num / den
    return pd.DataFrame(res, index=cols, columns=cols)

def pval_matrix(corr, df):
    n = len(df)
    cols = df.columns
    res = np.zeros((len(cols), len(cols)))
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j:
                res[i, j] = 0
                continue
            r = corr.iloc[i, j]
            if abs(r) == 1:
                res[i, j] = 0
                continue
            t_stat = r * np.sqrt((n - 2) / (1 - r**2))
            p = 2 * (1 - t.cdf(abs(t_stat), df=n-2))
            res[i, j] = p

    return pd.DataFrame(res, index=cols, columns=cols)

def pirson_visualization(df):
    corr = corr_matrix_pirson(df)
    pvals = pval_matrix(corr, df)
    
    plt.figure(figsize=(5, 5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".4f",
        cmap="coolwarm",
        center=0
    )
    plt.title("Корреляционная матрица Пирсона")
    plt.show()
    
    plt.figure(figsize=(5, 5))
    sns.heatmap(
        pvals,
        annot=True,
        fmt=".5f",
        cmap="coolwarm"
    )
    plt.title("P-values для парных коэффициентов Пирсона")
    plt.show()





## Матрица частных коэффициентов
def partial_correlation_matrix(data):
    cols = data.columns
    n = len(cols)
    corr_matrix = data.corr().values
    part_corr = np.zeros((n, n))
    part_p = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                part_corr[i, j] = 1.0
                part_p[i, j] = 0.0
            else:
                R = corr_matrix.copy()
                R_ii = np.delete(np.delete(R, i, axis=0), i, axis=1)
                R_jj = np.delete(np.delete(R, j, axis=0), j, axis=1)
                R_ij = np.delete(np.delete(R, i, axis=0), j, axis=1)
                sign = (-1) ** (i + j)
                det_R_ii = np.linalg.det(R_ii)
                det_R_jj = np.linalg.det(R_jj)
                det_R_ij = np.linalg.det(R_ij)
                part_corr[i, j] = -sign * det_R_ij / np.sqrt(det_R_ii * det_R_jj)
                dfree = len(data) - (n - 2) - 2
                if dfree > 0 and abs(part_corr[i, j]) < 1:
                    t_stat = part_corr[i, j] * np.sqrt(dfree / (1 - part_corr[i, j]**2))
                    part_p[i, j] = 2 * (1 - t.cdf(abs(t_stat), df=dfree))
                else:
                    part_p[i, j] = 1.0
                    
    return pd.DataFrame(part_corr, index=cols, columns=cols), pd.DataFrame(part_p, index=cols, columns=cols)

def partial_correlation_visualization(df):
    pc_matrix, pc_pvals = partial_correlation_matrix(df)
    plt.figure(figsize=(6, 6))
    sns.heatmap(pc_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0)
    plt.title("Матрица частных коэффициентов корреляции")
    plt.show()
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(pc_pvals, annot=True, fmt=".3f", cmap="coolwarm")
    plt.title("p-value для матрицц частных коэффициентов корреляции")
    plt.show()




### Матрица множественных коэффициентов корреляции
def multiple_correlation_det(data):
    cols = data.columns
    R_vals = pd.Series(index=cols, dtype=float)
    pval = pd.Series(index=cols, dtype=float)
    R = data.corr().values
    det_R = np.linalg.det(R)
    n = len(data)
    m = len(cols)
    for i, dep in enumerate(cols):
        R_ii = np.delete(np.delete(R, i, axis=0), i, axis=1)
        det_R_ii = np.linalg.det(R_ii)
        R_sq = 1 - det_R / det_R_ii
        R_vals[dep] = np.sqrt(R_sq)
        k = m - 1
        F = (R_sq / k) / ((1 - R_sq) / (n - k - 1))
        pval[dep] = 1 - f.cdf(F, k, n - k - 1)
    
    return R_vals, pval

def multiple_correlation_visualization(df):
    R_vals, pvals = multiple_correlation_det(df)
    
    plt.figure(figsize=(8, 2))
    sns.heatmap(R_vals.to_frame().T, annot=True, fmt=".3f", cmap="coolwarm", center=0)
    plt.title("Множественные коэффийиенты корреляции")
    plt.show()
    
    plt.figure(figsize=(8, 2))
    sns.heatmap(pvals.to_frame().T, annot=True, fmt=".3f", cmap="coolwarm")
    plt.title("р-values для множественных коэффициентов корреляции")
    plt.show()




### Матрица Спирмана
def spearman_matrix(df):
    cols = df.columns
    n = len(cols)
    corr = np.zeros((n, n))
    pvals = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i, j] = 1.0
                pvals[i, j] = 0.0
            else:
                x = rankdata(df.iloc[:, i])
                y = rankdata(df.iloc[:, j])
                r, p = spearmanr(x, y)
                corr[i, j] = r
                pvals[i, j] = p
    
    return pd.DataFrame(corr, index=cols, columns=cols), pd.DataFrame(pvals, index=cols, columns=cols)

def spearman_visualization(df):
    corr_s, pvals_s = spearman_matrix(df)

    plt.figure(figsize=(5, 5))
    sns.heatmap(corr_s, annot=True, fmt=".4f", cmap="coolwarm", center=0)
    plt.title("Матрица Спирмена")
    plt.show()
    
    plt.figure(figsize=(5, 5))
    sns.heatmap(pvals_s, annot=True, fmt=".5f", cmap="coolwarm")
    plt.title("p-values для матрицы Спирмена")
    plt.show()