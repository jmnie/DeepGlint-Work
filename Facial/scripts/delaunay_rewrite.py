from scipy.spatial import Delaunay
import numpy as np
from affectNet_sort import read_csv
import json

def delaunay_plot(points):

    #points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
    tri = Delaunay(points)

    print(tri)
    #import matplotlib.pyplot as plt
    #plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
    #plt.plot(points[:,0], points[:,1], 'o')
    #plt.show()

def test_plot():

    with open('training.json') as json_data:
        training_dict = json.load(json_data)

    test_key = "737db2483489148d783ef278f43f486c0a97e140fc4b6b61b84363ca.jpg"
    landmark = training_dict[test_key][4]
    landmark = np.array(landmark)
    delaunay_plot(landmark)

test_plot()