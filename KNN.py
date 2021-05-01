"""
生物信息学导论第七章作业
姓名：郭海鹏
作业内容：使用K近邻算法完成哺乳动物的分类器实现
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error


def knn(x_train, y_train,
        x_test, y_test):
    model = KNeighborsClassifier(n_neighbors=3)  # 经过测试,n_neighbors=3时，ROC曲线下面积最大=0.97，选择性最好
    model.fit(x_train, y_train)  # 训练数据，得到拟合的分类模型

    # 在训练集上检验预测效果
    predict_train = model.predict(x_train)
    print("         KNN         ")
    print("训练样本     :", y_train)
    print("训练集预测结果:", predict_train)
    print("训练集预测精确率", precision_score(y_train, predict_train, average='macro'), "\n")

    # 在测试集上检验预测效果
    predict_test = model.predict(x_test)
    print("测试标签:", y_test)
    print("预测结果:", predict_test, "\n")

    precision = precision_score(y_test, predict_test, average='macro')  # 精确率
    mean = mean_squared_error(y_test, predict_test)  # 均方误差

    print("测试集精确率  : ", precision)
    print("测试集均方误差: ", mean, "\n")

    return model.fit(x_train, y_train)  # 返回得到的拟合模型
