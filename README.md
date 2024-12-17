#applying pca algorithm to a dataset so that it's ready to be used later to train and evaluate algorithms

#import libraries
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#data separation into training set and testing set
mnist = fetch_openml('mnist_784')
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size = 1/7.0)

#fitting StandardScaler algorithm to the training set
scaler = StandardScaler()
scaler.fit(train_img)

#apply transform on both training set and testing set
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

#pca implementation
pca = PCA(0.95) #variation rate 95%
pca.fit(train_img)
pca.n_components_
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)
