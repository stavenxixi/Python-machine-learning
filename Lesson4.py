import numpy as np
import matplotlib.pyplot as plt
import mglearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier

#k近邻回归
mglearn.plots.plot_knn_regression(n_neighbors=1)

mglearn.plots.plot_knn_regression(n_neighbors=3)

#skleatn_k近邻回归
from sklearn.neighbors import KNeighborsRegressor
X, y = mglearn.datasets.make_wave(n_samples=40)
#将wave数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#模型实例化，并将邻居个数设为3
reg = KNeighborsRegressor(n_neighbors=3)
#利用数据和训练目标值来拟合模型
reg.fit(X_train, y_train)
#测试数据预测
print("Test set predictions:\n {}".format(reg.predict(X_test)))
#使用score来评估模型
print("Test set R^2 : {:.2f}".format(reg.score(X_test, y_test)))
fig, axes = plt.subplots(1,3, figsize=(15,4))
#创建1000个数据点，在-3和3之间均匀分布
line = np.linspace(-3, 3, 1000).reshape(-1,1)
for n_neighbors, ax in zip([1,3,9],axes):
    #利用1个、3个或9个邻居分别进行预测
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c = mglearn.cm2(0),markersize=8)
    ax.plot(X_test, y_test, 'v', c = mglearn.cm2(1), markersize=8)
    ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(n_neighbors, reg.score(X_train, y_train),
                                                                                 reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")
