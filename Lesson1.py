from sklearn.datasets import load_iris
import pandas as pd
import mglearn
import numpy as np

iris_dataset = load_iris()
print("Keys of iris_dataset:\n {}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names:{}".format(iris_dataset['target_names']))
print("Feature names: \n {}".format(iris_dataset['feature_names']))
print("Type of data:{}".format(type(iris_dataset['data'])))
print("Shape of data:{}".format(iris_dataset['data'].shape))
print("First fiev rows of data:\n{}".format(iris_dataset['data'][:5]))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Shape of target: {}".format(iris_dataset['target']))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0)
print("X_train shape:{}".format(X_train.shape))
print("y_train shape:{}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
X_new = np.array([[5,2.9,1,0.2]])
print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))
y_pred = knn.predict(X_test)
print("Test set predictions :\n {}".format(y_pred))
