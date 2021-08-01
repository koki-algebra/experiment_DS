import numpy as np
import matplotlib.pyplot as plt

### 多項式回帰に関する関数群 ###

# x座標のデータからデータ行列Xを生成する
def gen_data_matrix(data_x, M):
    X = np.empty((0, M + 1))

    for i in range(len(data_x)):
        xi = np.array([])
        for j in range(M + 1):
            xi = np.append(xi, data_x[i] ** j)
        X = np.append(X, np.array([xi]), axis=0)

    return X


# 訓練データdataでM次多項式回帰を行い、係数ベクトルを返す関数
def polynomial_regression(data, M):
    # x座標のデータ
    data_x = data[:, 0]
    # 教師データ
    t = data[:, 1]
    # 行列Xを作成
    X = gen_data_matrix(data_x, M)

    # 最小二乗法によって係数ベクトルを求める
    w = np.linalg.inv(X.T @ X) @ X.T @ t

    return w

# M次多項式回帰によって得られた係数ベクトルwを使って推定値y_eを返す関数
def predict(w, X):
    # 推定値を格納したベクトル
    y_e = np.array([])

    # 推定値を計算
    for i in range(X.shape[0]):
        y_e = np.append(y_e, np.dot(w, X[i]))
    
    return y_e


# 推定値y_e(ベクトル) と 教師データt(ベクトル)の平均二乗平方根誤差(Root Mean Square)
def calc_rms(y_e, t):
    N = len(t)
    return np.sqrt(np.dot((y_e - t).T, (y_e - t)) / N)