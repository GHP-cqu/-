"""
生物信息学导论第七章作业
姓名：郭海鹏
作业内容：使用K近邻算法 和 随机森林完成哺乳动物的分类器实现
"""
import numpy as np
import pandas as pd
import KNN
import ROC
import feature_importance as FI
import matplotlib.pyplot as plt
from RandomForest import RandomForestClassify as RFC

#       K近邻分类器
#       哺乳动物分类数据预处理
#     恒温 胎生 4条腿 冬眠 类标号
# 是    1   1   1    1    1
# 否    0   0   0    0    0

# 训练集
X_train = pd.DataFrame(np.array(
        [[1, 1, 1, 1],
         [1, 1, 1, 0],
         [1, 1, 0, 1],
         [1, 1, 0, 0],
         [0, 0, 1, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 0],
         [1, 0, 0, 0]]), columns=['体温', '胎生', '4条腿', '冬眠'])
Y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])

# 测试集
X_test = pd.DataFrame(np.array(
        [[1, 1, 0, 0],
         [1, 0, 0, 0],
         [1, 1, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 0, 1, 1]]), columns=['体温', '胎生', '4条腿', '冬眠'])

Y_test = np.array([1, 0, 1, 0, 0, 0, 1, 1])

# K近邻
knn = KNN.knn(X_train, Y_train, X_test, Y_test)  # 拟合后的knn模型
# 绘制ROC曲线
plt.figure(1)
ROC.roc(knn, X_test, Y_test)
plt.title("KNN ROC")
# 求特征贡献度
plt.figure(2)
FI.feature_importance(X_train, Y_train, knn, select=1)
plt.title("KNN 特征贡献度")

# 随机森林
RF = RFC(X_train, Y_train, X_test, Y_test)  # 拟合后的RandomForest模型
# 绘制ROC曲线
plt.figure(3)
ROC.roc(knn, X_test, Y_test)
plt.title("RandomForest ROC")
# 求特征贡献度
plt.figure(4)
FI.feature_importance(X_train, Y_train, RF, select=0)
plt.title("RandomForest 特征贡献度")

plt.show()
