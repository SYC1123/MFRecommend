import numpy as np


def gradAscent(dataMat, k, alpha, beta, maxCycles):
    '''
    利用梯度下降法对矩阵进行分解
    :param dataMat: （mat）用户商品矩阵
    :param k: （int）分解矩阵的参数
    :param alpha: （float）学习率
    :param beta:（float）正则化参数
    :param maxCycles: （int）最大迭代次数
    :return: p,q（mat）分解后的参数
    '''
    m, n = np.shape(dataMat)
    # 1. 初始化p和q
    p = np.mat(np.random.random((m, k)))  # 代表生成m行 k列的随机浮点数，浮点数范围 : (0,1)
    q = np.mat(np.random.random((k, n)))  # 代表生成k行 m列的随机浮点数，浮点数范围 : (0,1)

    # 2.开始训练
    for step in range(maxCycles):
        for i in range(m):
            for j in range(n):
                if dataMat[i, j] > 0:
                    error = dataMat[i, j]
                    for r in range(k):
                        error = error - p[i, r] * q[r, j]
                    for r in range(k):
                        # 梯度上升
                        p[i, r] = p[i, r] + alpha * (2 * error * q[r, j] - beta * p[i, r])
                        q[r, j] = q[r, j] + alpha * (2 * error * p[i, r] - beta * q[r, j])
        loss = 0.0
        for i in range(m):
            for j in range(n):
                if dataMat[i, j] > 0:
                    error = 0.0
                    for r in range(k):
                        error = error + p[i, r] * q[r, j]
                    # 3.计算损失函数
                    loss = (dataMat[i, j] - error) * (dataMat[i, j] - error)
                    for r in range(k):
                        loss = loss + beta * (p[i, r] * p[i, r] + q[r, j] * q[r, j]) / 2
        if loss < 0.001:
            break
        if step % 1000 == 0:
            print('titer:', step, "loss:", loss)
    return p, q


def prediction(dataMatrix, p, q, user):
    '''
    为用户user推荐未互动的项打分
    :param dataMatrix: （mat）原始用户商品矩阵
    :param p: （mat）分解后的矩阵p
    :param q: （mat）分解后的矩阵q
    :param user: （int）用户的id
    :return: predict（list）推荐列表
    '''
    n = np.shape(dataMatrix)[1]
    predict = {}
    for j in range(n):
        if dataMatrix[user, j] == 0:
            predict[j] = (p[user,] * q[:, j])[0, 0]

    # 按照打分从大到小排序
    return sorted(predict.items(), key=lambda d: d[1], reverse=True)


def load_data(path):
    '''
    导入数据
    :param path:（string）用户商品矩阵存储位置
    :return: data（mat）用户商品矩阵
    '''
    f = open(path)
    data = []
    for line in f.readlines():
        arr = []
        lines = line.strip().split(',')
        for x in lines:
            if x != '-':
                arr.append(float(x))
            else:
                arr.append(float(0))
        data.append(arr)
    f.close()
    return np.mat(data)


def save_file(file_name, source):
    '''
    保存结果
    :param file_name:（string）需要保存的文件名
    :param source: （mat）需要保存的文件
    :return:
    '''
    f = open(file_name, 'w')
    m, n = np.shape(source)
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append((str(source[i, j])))
        f.write('\t'.join(tmp) + '\n')
    f.close()


def top_k(predict, k):
    '''
    为用户推荐前K个商品
    :param predict: （list）排序好的商品列表
    :param k: （int）推荐的商品数量
    :return: top_recom（list）top_k个商品
    '''
    top_recom = []
    len_result = len(predict)
    if k >= len_result:
        top_recom = predict
    else:
        for i in range(k):
            top_recom.append(predict[i])
    return top_recom


# 由于训练时p，q矩阵是随机生成初始化的，所以导致每一次的损失loss都不同

if __name__ == '__main__':
    # 1.导入用户商品矩阵
    print('--------1.load data-------')
    dataMatrix = load_data('data.txt')
    # 2.利用梯度下降法对矩阵进行分解
    print('--------2.training -------')
    p, q = gradAscent(dataMatrix, 5, 0.0002, 0.02, 5000)
    # 3.保存分析后的结果
    print('--------3.save decompose--')
    save_file('p', p)
    save_file('q', q)
    # 4.预测
    print('--------4.prediction -----')
    predict = prediction(dataMatrix, p, q, 0)
    # 5.进行top_k推荐
    print('--------5.top_k recommendation ----')
    top_recom = top_k(predict, 2)
    print(top_recom)
    # 训练1
    '''
    titer: 0 loss: 18.744814426162165
    titer: 1000 loss: 0.9828723400159889
    titer: 2000 loss: 0.2695648538455083
    titer: 3000 loss: 0.1111535239592657
    titer: 4000 loss: 0.10232068753896116
    --------3.save decompose--
    --------4.prediction -----
    --------5.top_k recommendation ----
    [(2, 3.9220725456436645), (4, 3.8154560479653696)]
    '''
    # 训练2
    '''
    
    titer: 0 loss: 9.482553076210161
    titer: 1000 loss: 1.135357414330814
    titer: 2000 loss: 0.5352872387780825
    titer: 3000 loss: 0.15139304827215333
    titer: 4000 loss: 0.10692710770766595
    --------3.save decompose--
    --------4.prediction -----
    --------5.top_k recommendation ----
    [(4, 3.460306894054137), (2, 2.7461750333846786)]
    '''
# 两次训练之后得到的推荐不同？
# 原因是p,q分解矩阵随机给矩阵赋值的时候导致不同计算结果
# 训练结果不同所以需要训练多次然后取打分的平均值
