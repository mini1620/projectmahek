import pandas as pd
import matplotlib.pyplot as mtp
dataset=pd.read_csv('mall.csv')
x=dataset.iloc[:,[3,4]].values
import scipy.cluster.hierarchy as shc
dendro=shc.dendrogram(shc.linkage(x, method="ward"))
mtp.title("Dendrom Plot")
mtp.xlabel("Eucliddean distance")
mtp.ylabel("customers")
mtp.show()
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='eucliden',linkage='ward')
y_pred=hc.fit_predict(x)
mtp.scatter(x[y_pred == 0,0], x[y_pred == 0, 1], s=100, c='Blue', label='cluster 1')
mtp.scatter(x[y_pred == 1,0], x[y_pred == 1, 1], s=100, c='Green', label='cluster 2')
mtp.scatter(x[y_pred == 2,0], x[y_pred == 2, 1], s=100, c='red', label='cluster 3')
mtp.scatter(x[y_pred == 3,0], x[y_pred == 3, 1], s=100, c='cyan', label='cluster 4')
mtp.scatter(x[y_pred == 4,0], x[y_pred == 4, 1], s=100, c='magneta', label='cluster 5')
mtp.title('Cluster of customers')
mtp.xlabel('Annual income(k$')
mtp.ylabel('Spending Score (1-100)')
mtp.legend()
mtp.show()