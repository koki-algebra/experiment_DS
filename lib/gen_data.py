import numpy as np

# N個の(x, y)の組をcsvファイルに書き込む関数
def gen_data(N):
    # ndarray初期化
    data = np.empty((0, 2))

    # x,y座標生成
    x = np.array(np.random.rand(N))
    y = np.sin(2 * np.pi * x)

    noise = np.random.normal(0, 0.1, N)

    # y座標にnoiseを乗せる
    y += noise

    # dataに座標データを入れる
    for i in range(len(x)):
        data = np.append(data, np.array([[x[i], y[i]]]), axis=0)

    # csvファイルに書き込み
    np.savetxt(
        fname='./data/data_{N}.csv'.format(N=N),
        X=data,
        delimiter=','
    )