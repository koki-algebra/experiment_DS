import numpy as np
import matplotlib.pyplot as plt
from lib import polynomial_regression as pr

# 2日目
# 課題7.9用
def prob7_9(M, data_size):
    data = np.loadtxt('data/data_{N}.csv'.format(N=data_size), delimiter=',')

    x = data[:, 0]
    y = data[:, 1]

    X = pr.gen_data_matrix(x, M)

    w = pr.polynomial_regression(data, M)

    y_e = pr.predict(w, X)

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


# 3日目
# データサイズと次数に対応する訓練誤差とテスト誤差を返す
def prob_day3_calc_err(data_size, degree):
    data = np.loadtxt('data/data_{N}.csv'.format(N=data_size), delimiter=',')
    test_data_size = int(data_size / 4)
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
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, ax = plt.subplots()

    ax.scatter(x, train_errs, label='training error', s=5)
    ax.scatter(x, test_errs, label='test error', s=5)

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
        errors = np.append(errors, pr.cross_validation(data, 5, i))
    print(errors)

    fig, ax = plt.subplots()

    ax.scatter(range(1, 10), errors)
    ax.set_title('5 fold cross-validation')
    ax.set_xlabel('degree')
    ax.set_ylabel('test error')
    ax.grid()

    fig.savefig('img/day4/cv.png')
    plt.show()

def prob7_21_22():
    data = np.loadtxt('data/data_20.csv', delimiter=',')

    # モデル選択
    M = pr.select_best_degree(data)

    # 求めた次数Mを使って多項式回帰
    X = pr.gen_data_matrix(data[:,0], M)
    w = pr.polynomial_regression(data, M)
    y_e = pr.predict(w, X)

    fig, ax = plt.subplots()

    ax.scatter(data[:,0], data[:,1], label='correct answer', s=5)
    ax.scatter(data[:,0], y_e, label='predicted', s=5)
    ax.set_title('model select')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    ax.legend()

    fig.savefig('img/day4/model_select.png')
    plt.show()