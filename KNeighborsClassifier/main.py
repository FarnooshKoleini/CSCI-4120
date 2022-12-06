from socketserver import DatagramRequestHandler
import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def loadDataset(filename, split):
	df = pd.read_csv(filename, header=None)
	array = df.to_numpy()
	random.shuffle(array)
	training_len = int(len(array)*split)
	return array
	
def main():
	
	# prepare data
	split = 0.67
	url = 'https://raw.githubusercontent.com/ruiwu1990/CSCI_4120/master/KNN/iris.data'
	dataSet = loadDataset(url, split)

	accuracies = []
	# arbitrary value, won't be used in any calculations or be shown on the graph, just used to get every 
	# value in their correct poistions in the list according to the k value it represents
	accuracies.append(100)
	for k in range(1, 21):
		print("k:", k)
		accuracy = 0
		for i in range(5):
			Xtrain, Xtest, y_train, y_test = train_test_split(dataSet[:, :4], dataSet[:, 4], train_size = 0.67)
			knn = KNeighborsClassifier(k)
			knn.fit(Xtrain, y_train)
			predict = knn.predict(Xtest)
			accuracy += (y_test == predict).mean()
		accuracy = (accuracy/5) * 100
		accuracies.append(accuracy)
		print('Accuracy:', accuracy)

	plt.xlabel('k-value') 
	plt.ylabel('Accuracy (%)')
	ax=plt.subplot(111)
	ax.set_xlim(1, 21)
	ax.set_ylim(min(accuracies)-1, 101)
	dim=np.arange(1,21,1)
	ax.plot(accuracies, color='r',linewidth=1.0, label="Graph2")
	plt.xticks(dim)
	plt.grid()   
	plt.show()    
	plt.close()
	

main()
