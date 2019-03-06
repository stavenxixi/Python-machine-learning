from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn
from sklearn.preprocessing import MinMaxScaler

#构造数据
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)

#将其分为训练集和测试集
X_train, X_test = train_test_split(X, random_state=5, test_size=.1)

#绘制训练集和测试集
fig, axes = plt.subplots(1,3,figsize=(13,4))
axes[0].scatter(X_train[:,0], X_train[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^', c=mglearn.cm2(1), label="Test set", s=60)
axes[0].legend(loc="Upper left")
axes[0].set_title("Original data")

#利用MinMaxScaler 缩放数据abs
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#将正确缩放的数据可视化
axes[1].scatter(X_train_scaled[:,0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', c=mglearn.cm2(1), label="Test set", s=60)
axes[1].set_title("Scaled data")

#单独对测试集进行缩放
#使得测试集的最小值为0，最大值为1
#千万不要这么做！这里只是为了举例
test_scaler = MinMaxScaler()
test_scaler.fit(X_test)
X_test_scaled_badly = test_scaler.transform(X_test)

#将错误缩放的数据可视化
axes[2].scatter(X_train_scaled[:,0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1], marker='^', c=mglearn.cm2(1), label="Test set", s=60)
axes[2].set_title("Improperly Scaled data")

for ax in axes:
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
