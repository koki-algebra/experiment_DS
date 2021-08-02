import numpy as np

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
    # データ行列Xを作成
    X = gen_data_matrix(data_x, M)

    # 最小二乗法によって係数ベクトルを求める
    w = np.linalg.inv(X.T @ X) @ X.T @ t

    return w


# 正則化多項式回帰を行う関数(L: 正規化パラメータλ)
def normalize_pr(data, degree, L):
    # x座標のデータ
    data_x = data[:,0]
    # 教師データ
    t = data[:, 1]
    # データ行列Xを作成
    X = gen_data_matrix(data_x, degree)

    # 最小二乗法によって係数ベクトルを求める
    w = np.linalg.inv(X.T @ X + np.identity(degree + 1) * L) @ X.T @ t
    return np.array(w)


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


# N個の1〜Kの乱数を生成する関数
def gen_random_nums(N, K):
    rnds = np.array([])
    # 1〜Kの全ての数字が出たか
    is_full_exist = False

    while not is_full_exist:
        rnds = np.array([])
        for i in range(N):
            rnds = np.append(rnds, np.random.randint(1, K + 1))

        # 出ていない数字がないか確認
        for i in range(1, K + 1):
            if not i in rnds:
                is_full_exist = False
                break
            else: # 出ていない数字が無い場合ループを抜ける
                is_full_exist = True

    return rnds


# K-Fold Cross Validation (type: 'degree' or 'lambda')
def cross_validation(data, K, type, degree, L=0):
    data_size = data.shape[0]
    rnds = gen_random_nums(data_size, K)

    errors = np.array([])
    for i in range(1, K + 1):
        # テストデータ
        d_test = data[rnds == i]
        # 訓練データ
        d_train = data[rnds != i]

        # 訓練データを用いて係数ベクトルを求める
        if type == 'degree':  # 次数に関する交差検証
            w = polynomial_regression(d_train, degree)
        elif type == 'lambda':  # 正則化パラメータに関する交差検証
            w = normalize_pr(data, degree, L)
        else:
            print('select degree or lambda')
            return

        # テストデータのデータ行列
        X_test = gen_data_matrix(d_test[:,0], degree)
        # テストデータに対する予測値を計算
        y_e = predict(w, X_test)
        # テスト誤差を計算し配列に格納
        errors = np.append(errors, calc_rms(y_e, d_test[:,1]))

    # K通りのテスト誤差の平均を返す
    return np.mean(errors)


# 交差検証の結果最もテスト誤差の小さかった次数を選択する関数
def select_best_degree(data):
    errors = np.array([])

    for i in range(1, 10):
        errors = np.append(errors, cross_validation(data, 5, 'degree', i))

    # 誤差が一番小さかった次数を返す
    return np.argmin(errors) + 1

def select_best_lambda(data, L_list):
    errors = np.array([])

    for i in range(len(L_list)):
        errors = np.append(errors, cross_validation(data, 5, 'lambda', 9, L_list[i]))
    
    # 誤差が一番小さかった正則化パラメータを返す
    return L_list[np.argmin(errors)]