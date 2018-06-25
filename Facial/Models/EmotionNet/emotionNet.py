import numpy as np
import math 
import itertools  
from scipy.spatial import Delaunay
import json 

def distance(A,B):
    dis = math.sqrt(math.pow(A[0] - B[0],2) + math.pow(A[1] - B[1],2))
    return dis

def dotProduct(A,B,C):
    res = np.dot( np.array([C[0]-A[0],C[1]-A[1]] ), np.array([B[0]-A[0],B[1]-A[1]] ) )
    return res

def getAngle(A,B,C):
    # Point A, B, C
    # calculate the angle A,B,C
    #Angle A:  
    angleA = np.arccos(dotProduct(A,B,C)/(distance(A,B)*distance(A,C)))
    angleB = np.arccos(dotProduct(B,A,C)/(distance(A,B)*distance(B,C)))
    angleC = np.arccos(dotProduct(C,B,A)/(distance(C,B)*distance(A,C)))
    return angleA,angleB,angleC

def featureVector(point_array):
    #points = np.array(point_array)
    tri = Delaunay(point_array)
    fv1 = []
    fv2 = []

    coor_list = list(itertools.combinations(point_array,2))

    for ele in coor_list:
        d = distance(ele[0],ele[1])
        fv1.append(d)

    for ele in tri.simplices:
        a1,a2,a3 = getAngle(point_array[ele[0]],point_array[ele[1]],point_array[ele[2]])
        fv2.append(a1)
        fv2.append(a2)
        fv2.append(a3)

    # type: list
    mergelist = fv1 + fv2
    return mergelist 

def gabor_filter(landmark):
    

#def KSDA():

if __name__ == '__main__':

    with open('/home/jiaming/code/github/DeepGlint-Work/Facial/Models/EmotionNet/vali.json','r') as fp:
        vali_dict = json.load(fp)

    for key in vali_dict:
        landmark = vali_dict[key][1]
        break

    print("Length of points",len(landmark))
    res = featureVector(landmark)
    print(len(res))

