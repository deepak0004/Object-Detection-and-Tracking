from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import random
import pickle

digits = datasets.load_digits()
flatten = []

'''
for eachDigit in digits['images']:
    temp = []
    for eachrow in eachDigit:
    	for val in eachrow:
    	   dig = random.randint(1, 16)
           temp.append( float(dig) )
    flatten.append(temp)
'''
for eachDigit in digits['images']:
    temp = []
    for eachrow in eachDigit:
         temp.extend( eachrow )
    #print temp
    #exit()
    flatten.append(temp) 

labll = digits['target']
X_tsne = TSNE(learning_rate=100).fit_transform(flatten)
plt.figure(figsize=(10, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=digits['target'], cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.savefig('zdim.png')

with open('label_space' + '.dump', "wb") as fp: 
    pickle.dump(flatten, fp)