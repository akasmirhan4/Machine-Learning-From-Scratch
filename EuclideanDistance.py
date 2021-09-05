from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
style.use('fivethirtyeight')

# euclideanDistance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]

# [[plt.scatter(ii[0], ii[1], s=100, color=i)
#   for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], s = 100, color = "g")
# plt.show()


def KNN(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')

    distances = []
    for group in data:
        for features in data[group]:
            euclideanDistance = np.linalg.norm(
                np.array(features) - np.array(predict))
            distances.append([euclideanDistance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    votes_result = Counter(votes).most_common(1)[0][0]
    return votes_result

result = KNN(dataset, new_features, k=3)
print(result)