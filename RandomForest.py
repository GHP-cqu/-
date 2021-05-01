"""
生物信息学导论第七章作业
姓名：郭海鹏
作业内容：使用随机森林算法完成哺乳动物的分类器实现
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error


def RandomForestClassify(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(criterion='entropy', min_samples_split=3)
    model.fit(x_train, y_train)  # 训练数据，得到拟合的分类模型

    # 在训练集上检验预测效果
    predict_train = model.predict(x_train)
    print("    RandomForest    ")
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
