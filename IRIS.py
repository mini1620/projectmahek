from sklearn.datasets import load_iris
from sklearn import preprocessing
iris = load_iris()
print(iris.data.shape)
x = iris.data
y = iris.data
normalized_x = preprocessing.normalize(x)
