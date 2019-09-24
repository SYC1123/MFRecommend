import numpy as np
from mf import load_data, save_file, prediction, top_k


def train(V, r, maxCycles, e):
    '''
    非负矩阵分解
    :param V: （mat）评分矩阵
    :param r: 分解后矩阵的维数
    :param maxCycles: 最大的迭代次数
    :param e: （int）最大的迭代次数
    :return: W，H（mat）分解后的矩阵
    '''
    m, n = np.shape(V)
    # 1.初始化矩阵
    W = np.mat(np.random.random((m, r)))
    H = np.mat(np.random.random((r, n)))

    # 2.非负矩阵分解
    for step in range(maxCycles):
        V_pre = W * H
        E = V - V_pre
        err = 0.0
        for i in range(m):
            for j in range(m):
                err += E[i, j] * E[i, j]

        if err < e:
            break
        if step % 1000 == 0:
            print('titer:', step, 'loss:', err)

        a = W.T * V
        b = W.T * W * H
        for i_1 in range(r):
            for j_1 in range(n):
                if b[i_1, j_1] != 0:
                    H[i_1, j_1] = H[i_1, j_1] * a[i_1, j_1] / b[i_1, j_1]

        c = V * H.T
        d = W * H * H.T
        for i_2 in range(m):
            for j_2 in range(r):
                if d[i_2, j_2] != 0:
                    W[i_2, j_2] = W[i_2, j_2] * c[i_2, j_2] / d[i_2, j_2]

    return W, H


if __name__ == '__main__':
    # 1.导入用户商品矩阵
    print('--------1.laod data--------')
    V = load_data("data.txt")
    # 2.非负矩阵分解
    print('--------2.training---------')
    W, H = train(V, 5, 10000, 1e-5)
    # 3.保存分解后的结果
    print('--------3.save decompose---')
    save_file('W', W)
    save_file('H', H)
    # 4.预测
    print('--------4.prediction-------')
    predict = prediction(V, W, H, 0)
    # 进行Top_K推荐
    print('--------5.top_k recommend--')
    top_recom = top_k(predict, 2)
    print(top_recom)
