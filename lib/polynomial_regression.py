import numpy as np
import matplotlib.pyplot as plt

### 多項式回帰に関する関数群 ###

# M次多項式回帰を行い、係数ベクトルを返す関数
def polynomial_regression(data, M):
    # x座標のデータ
    data_x = data[:, 0]
    # 教師データ
    t = data[:, 1]
    # 行列Xを初期化
    X = np.empty((0, M + 1))

    # 行ベクトルxiを作り、Xに加えていく
    for i in range(len(data_x)):
        xi = np.array([])
        for j in range(M + 1):
            xi = np.append(xi, data_x[i] ** j)
        X = np.append(X, np.array([xi]), axis=0)

    # 最小二乗法によって係数ベクトルを求める
    w = np.linalg.inv(X.T @ X) @ X.T @ t

    # 推定値を格納したベクトル
    y_e = np.array([])

    # 推定値を計算
    for i in range(len(data_x)):
        y_e = np.append(y_e, np.dot(w, X[i]))

    return y_e

# 通常の多項式回帰の結果をプロットする関数
def poly_reg_plot(M, data_size):
    data = np.loadtxt('data/data_{N}.csv'.format(N=data_size), delimiter=',')

    x = data[:, 0]
    y = data[:, 1]

    y_e = polynomial_regression(data, M)

    fig, ax = plt.subplots()

    ax.scatter(x, y, label='correct answer', s=5)
    ax.scatter(x, y_e, label='predicted', s=5)

    ax.set_title('Polynomial fitting (data:{N}, degree:{M})'.format(N=data_size, M=M))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.grid()
    ax.legend()

    fig.savefig('img/day2/fit_data{N}_deg{M}.png'.format(N=data_size, M=M))
    plt.show()