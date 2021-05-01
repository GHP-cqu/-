import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def feature_importance(x_train, y_train, model, select=None):
    # para:
    # select:因为使用随机森林时, 函数库中有自己的特征重要性函数feature_importances_
    #        而K近邻没有自带库中的特征重要性函数，所以使用通用的特征重要性函数
    if select == 1:
        results = permutation_importance(model, x_train, y_train, scoring='accuracy')  # 特征重要性排列
        features = np.array(x_train.columns)
        importance = results.importances_mean  # 特征重要性的平均值
        indices = np.argsort(importance)[::-1]  # 将特征重要性的值按从大到小排列，并将每个值在原数组位置，按排列后顺序赋给indices
    else:
        features = np.array(x_train.columns)
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
    # for f, i in zip(features, importance):
    #     print('Feature: %s, Score: %.5f' % (f, i))  # 打印特征及其对应贡献度打分
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使画图能显示中文标题
    plt.rcParams['axes.unicode_minus'] = False

    # 画特征贡献度的柱状图
    plt.title("特征贡献度")
    plt.bar(range(len(importance)), importance[indices])  # importance[indices]即将特征贡献度在坐标轴从大至小表示
    plt.xticks(range(len(importance)), [features[i] for i in indices])  # [features[i] for i in indices特征贡献度对应的特征
    for x, y in zip(range(len(importance)), importance[indices]):
        plt.text(x, y + 0.001, round(y, 4), ha='center')
    # plt.show()
