from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy.optimize import linear_sum_assignment as linear_assignment

# KMeans predictions, since its meant to be unsupervised learning, can not use normal ways of accurately measuring its accuracy
# because of this, the function below helps us with that by mapping the labels predicted by the kmeans algorithm to the true they best correspond with.
# This is best explained here https://datascience.stackexchange.com/a/64208
def getLabels(y_true, y_pred, k):
    values = np.zeros((k,k))
    for i in range(y_true.size):
        values[y_true[i]][y_pred[i]] += 1

    map = linear_assignment(values.max()-values)
    for j in range(y_pred.size):
        y_pred[j] = map[1][y_pred[j]]
    
    return y_pred


# Get the data
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

# Show the Elbow Visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,15))

visualizer.fit(X)
visualizer.show() 

k = visualizer.elbow_value_

# Use the best K to create a model
model = KMeans(k, random_state=0)
model.fit(X, y_true)
labels = model.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50, cmap='viridis')
plt.show()

# Use the function above to get a true representation of predicted values and measure accuracy
pred = getLabels(y_true, labels, k)
print("Accuracy Score:", accuracy_score(y_true, pred))

# Create heat map
sns.set()

mat = confusion_matrix(y_true, pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.show()
