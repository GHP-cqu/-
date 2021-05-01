import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc

# 计算ROC


def roc(model, x_test, y_test):
    y_positive_proba = model.predict_proba(x_test)[::, 1]  # 正标签的概率估计
    fpr, tpr, _ = roc_curve(y_test, y_positive_proba)  # 计算真正率:TPR; 假正率:FPR
    auc = roc_auc_score(y_test, y_positive_proba)  # 计算ROC曲线下的面积

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使画图能显示中文标题
    plt.rcParams['axes.unicode_minus'] = False

    # 画ROC曲线
    plt.plot(fpr, tpr, label=' AUC %0.2f'  % auc, color='blue', lw=2)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('特异度 1-Specificity')
    plt.ylabel('灵敏度 Sensitivity')
    plt.title('Receiver Operating Curve')
    plt.legend(loc="lower right")
    # plt.show()
