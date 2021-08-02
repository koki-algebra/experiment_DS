import numpy as np
import matplotlib.pyplot as plt
from lib import polynomial_regression as pr

# 2日目
# 課題7.9用
def prob7_9(M, data_size):
    data = np.loadtxt('data/data_{N}.csv'.format(N=data_size), delimiter=',')

    data_x = data[:, 0]
    data_y = data[:, 1]

    x = np.linspace(0, 1, 500)

    X = pr.gen_data_matrix(x, M)

    w = pr.polynomial_regression(data, M)

    y_e = pr.predict(w, X)

    fig, ax = plt.subplots()

    ax.scatter(data_x, data_y, label='correct answer', s=5)
    ax.plot(x, y_e, label='predicted', color='#ff4500')

    ax.set_title('Polynomial fitting (data:{N}, degree:{M})'.format(N=data_size, M=M))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.grid()
    ax.legend()

    fig.savefig('img/day2/7_9_data{N}_deg{M}.png'.format(N=data_size, M=M))
    plt.show()


# 3日目
# データサイズと次数に対応する訓練誤差とテスト誤差を返す
def prob_day3_calc_err(data_size, degree):
    data = np.loadtxt('data/data_{N}.csv'.format(N=data_size), delimiter=',')
    test_data_size = int(data_size / 2)
    train_data_size = int(data_size - test_data_size)
    d_train = data[:train_data_size]
    d_test = data[test_data_size:]

    # 多項式回帰
    w = pr.polynomial_regression(d_train, degree)

    # 訓練データ行列
    X_train = pr.gen_data_matrix(d_train[:,0], degree)
    # テストデータ行列
    X_test = pr.gen_data_matrix(d_test[:,0], degree)

    # 推定値を計算
    y_e_train = pr.predict(w, X_train)
    y_e_test = pr.predict(w, X_test)

    # 誤差を計算
    train_err = pr.calc_rms(y_e_train, d_train[:,1])
    test_err = pr.calc_rms(y_e_test, d_test[:,1])

    return train_err, test_err

# 課題7,15, 7.16用. データサイズを渡して実行すると次数を1〜9と変化させた際の訓練誤差とテスト誤差がグラフにプロットされる.
def prob7_15(data_size):
    train_errs = np.array([])
    test_errs = np.array([])

    # 次数1〜9における誤差を算出
    for i in range(1, 10):
        train_err, test_err = prob_day3_calc_err(data_size, i)
        train_errs = np.append(train_errs, train_err)
        test_errs = np.append(test_errs, test_err)

    # グラフに誤差をプロット
    fig, ax = plt.subplots()

    ax.scatter(range(1, 10), train_errs, label='training error', s=5)
    ax.scatter(range(1, 10), test_errs, label='test error', s=5)

    ax.set_title('Root mean square (data:{N})'.format(N=data_size))
    ax.set_xlabel('degree')
    ax.set_ylabel('root mean square')
    ax.legend()
    ax.grid()
    
    fig.savefig('img/day3/rms_data{N}.png'.format(N=data_size))
    plt.show()

def prob7_20():
    data = np.loadtxt('data/data_20.csv', delimiter=',')
    # 次数1〜9における交差検証の結果を格納する配列
    errors = np.array([])

    for i in range(1, 10):
        errors = np.append(errors, pr.cross_validation(data, 5, 'degree', i))

    fig, ax = plt.subplots()

    ax.scatter(range(1, 10), errors, s=10)
    ax.set_title('5 fold cross-validation')
    ax.set_xlabel('degree')
    ax.set_ylabel('test error')
    ax.grid()

    fig.savefig('img/day4/7_20.png')
    plt.show()

def prob7_21_22():
    data = np.loadtxt('data/data_20.csv', delimiter=',')
    data_x = data[:,0]
    data_y = data[:,1]

    x = np.linspace(0, 1, 500)

    # モデル選択
    M = pr.select_best_degree(data)

    # 求めた次数Mを使って多項式回帰
    X = pr.gen_data_matrix(x, M)
    w = pr.polynomial_regression(data, M)
    y_e = pr.predict(w, X)

    fig, ax = plt.subplots()

    ax.scatter(data_x, data_y, label='correct answer', s=5)
    ax.plot(x, y_e, label='predicted', color='#ff4500')
    ax.set_title('model select (degree: {M})'.format(M=M))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.grid()
    ax.legend()

    fig.savefig('img/day4/7_22.png')
    plt.show()

# 5日目
# 正則化パラメータLで正則化多項式回帰を行い、訓練誤差とテスト誤差を返す
def prob_day5_calc_err(L):
    data = np.loadtxt('data/data_20.csv', delimiter=',')
    d_train = data[:10]
    d_test = data[10:]

    w = pr.normalize_pr(d_train, 9, L)

    X_train = pr.gen_data_matrix(d_train[:,0], 9)
    X_test = pr.gen_data_matrix(d_test[:, 0], 9)

    y_e_train = pr.predict(w, X_train)
    y_e_test = pr.predict(w, X_test)

    train_err = pr.calc_rms(y_e_train, d_train[:,1])
    test_err = pr.calc_rms(y_e_test, d_test[:,1])

    return train_err, test_err, w

def prob7_25():
    train_errs = np.array([]) # 訓練誤差
    test_errs = np.array([]) # テスト誤差
    w_norms = np.array([]) # 係数ベクトルのノルム
    L_list = np.array([]) # 正則化パラメータ

    for i in range(13):
        L_list = np.append(L_list, (1 / np.power(10, 8)) * np.power(10, i))

    for i in range(len(L_list)):
        # 正規化パラメータλ
        # 係数ベクトルw, 訓練誤差, テスト誤差を求める
        train_err, test_err, w = prob_day5_calc_err(L_list[i])
        train_errs = np.append(train_errs, train_err)
        test_errs = np.append(test_errs, test_err)
        # wのノルムを配列に格納
        w_norms = np.append(w_norms, np.linalg.norm(w))

    # グラフにプロット
    fig, axes = plt.subplots(1, 2)

    axes[0].scatter(L_list, train_errs, label='train error', s=10)
    axes[0].scatter(L_list, test_errs, label='test error', s=10)
    axes[1].scatter(L_list, w_norms, s=10)

    axes[0].set_title('errors')
    axes[0].set_xlabel('Regularization parameters')
    axes[0].set_ylabel('error')
    axes[0].set_xscale('log')
    axes[0].set_xlim(L_list[0], L_list[len(L_list) - 1])
    axes[0].legend()

    axes[1].set_title('weight vector')
    axes[1].set_xlabel('Regularization parameters')
    axes[1].set_ylabel('weight vector norm')
    axes[1].set_xscale('log')
    axes[1].set_xlim(L_list[0], L_list[len(L_list) - 1])

    fig.savefig('img/day5/7_25.png')
    plt.tight_layout()
    plt.show()

def prob7_26():
    data = np.loadtxt('data/data_20.csv', delimiter=',')

    L_list = np.array([])
    errors = np.array([])

    for i in range(13):
        L_list = np.append(L_list, (1 / np.power(10, 8)) * np.power(10, i))
    
    for i in range(len(L_list)):
        errors = np.append(errors, pr.cross_validation(data, 5, 'lambda', 9, L_list[i]))
    
    fig, ax = plt.subplots()

    ax.scatter(L_list, errors, s=10)
    ax.set_title('cv about normalization params')
    ax.set_xlabel('normalization parameters')
    ax.set_ylabel('test error')
    ax.set_xscale('log')
    ax.set_xlim(L_list[0], L_list[len(L_list) - 1])
    ax.grid()

    fig.savefig('img/day5/7_26.png')
    plt.show()

def prob7_27_28():
    data = np.loadtxt('data/data_20.csv', delimiter=',')

    # x,y座標のデータ
    data_x = data[:,0]
    data_y = data[:,1]

    L_list = np.array([])
    for i in range(13):
        L_list = np.append(L_list, (1 / np.power(10, 8)) * np.power(10, i))
    
    L = pr.select_best_lambda(data, L_list)
    print('optimal lambda:', L)


    x = np.linspace(0, 1, 500)
    X = pr.gen_data_matrix(x, 9)

    # 係数ベクトルw
    w = pr.normalize_pr(data, 9, L)

    # 多項式のy座標
    y_e = pr.predict(w, X)

    # グラフにプロット
    fig, ax = plt.subplots()

    ax.scatter(data_x, data_y, label='correct answer', s=10)
    ax.plot(x, y_e, label='predict', color='#ff4500')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.grid()
    ax.legend()

    fig.savefig('img/day5/7_28.png')
    plt.show()