import os 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import np_utils
from sklearn.neural_network import MLPClassifier

def transferLabel(list_train):
    label_list = []
    for i in range(len(list_train)):
        temp = list_train[i][-1]
        label_list.append(temp)
    label_ = set(label_list)
    return label_

def convertStringToFloat(list_):
    for i in range(len(list_)):
        list_[i] = float(list_[i])
    return list_

def transferLabelToNumber(label_):
    #'smurf', 'neptune', 'satan.', 'ipsweep', 'satan', 'normal', 'back'
    new_label = [0] * len(label_)

    for i in range(len(label_)):
        if label_[i] == "smurf":
            new_label[i] = 1
        elif label_[i] == "neptune":
            new_label[i] = 2
        elif label_[i] == "satan":
            new_label[i] = 3
        elif label_[i] == "ipsweep":
            new_label[i] = 4
        elif label_[i] == "normal":
            new_label[i] = 5
        else:
            new_label[i] = 6
    new_label = np.array(new_label)
    return new_label

def ReadData():
    current_path = os.path.dirname(__file__)
    train_path = current_path + '/data/intrusion_train_raw.txt'
    test_path = current_path + '/data/intrusion_test_raw.txt'

    file_train = open(train_path,"r")
    file_test = open(test_path,"r")

    list_train = []
    list_test = []
    y_train = []
    y_test = []

    for line in file_train:
        fileds = line.split(",")
        fileds[-1] = fileds[-1].replace(".\n","")
        y_train.append(fileds[-1])
        temp_fileds = fileds[4:41]
        temp_fileds = convertStringToFloat(temp_fileds)
        list_train.append(temp_fileds)
    
    for line in file_test:
        fileds = line.split(",")
        fileds[-1] = fileds[-1].replace(".\n","")
        y_test.append(fileds[-1])
        temp_fileds = fileds[4:41]
        temp_fileds = convertStringToFloat(temp_fileds)
        list_test.append(temp_fileds)
    
    y_train = transferLabelToNumber(y_train)
    y_test = transferLabelToNumber(y_test)
    x_train = np.array(list_train)
    x_test = np.array(list_test)

    return x_train,y_train,x_test,y_test

    # Read .txt file line by line seperated by semicolon 

x_train,y_train,x_test,y_test = ReadData()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


def getAccuracy(result,y_test):
    correct = 0
    for i in range(len(result)):
        if y_test[i] == result[i]:
            correct = correct +1

    accuracy = (correct/float(len(y_test)))*100
    return accuracy

# Random Forest
def RandomForest_(train_data,train_label,test_data,test_label):
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(train_data, train_label)
    result = clf.predict(test_data)
    acc_ = getAccuracy(result,test_label)
    return acc_

## KNN
def knn_(train_data,train_label,test_data,test_label):
        #On the following is the KNN algorithm:
    k = 10 # 1600*10 = 160000

    predictions = []

    for ele_test in test_data:
        distance = []
        for ele_train in train_data:
            temp_distance = np.sum(np.power(ele_test-ele_train,2))
            distance.append(temp_distance)

        distance = np.array(distance)
        sortdiffidx = distance.argsort()

        vote = {}  # create the dictionary
        for i in range(k):
            ith_label = train_label[sortdiffidx[i]]
            vote[ith_label] = vote.get(ith_label,0) + 1  # get(ith_label,0) : if dictionary 'vote' exist key 'ith_label', return vote[ith_label]; else return 0
        sortedvote = sorted(vote.items(), key=lambda x: x[1], reverse=True)
        # 'key = lambda x: x[1]' can be substituted by operator.itemgetter(1)
        predictions.append(sortedvote[0][0])

    test_label = test_label.tolist()
    accu = getAccuracy(predictions,test_label)
    return accu

## Decision Tree
def decision_tree(train_data,train_label,test_data,test_label):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data,train_label)
    result = clf.predict(test_data)
    accu = getAccuracy(result,test_label)

    return accu
## SVM 
def svm_(X_train, y_train, X_test, y_test):
    clf = svm.SVC(C=1.0, cache_size=500, class_weight=None, coef0=0.0,
        decision_function_shape='ovo', degree=9, gamma='auto', kernel='rbf',
        max_iter=-1, probability=True, random_state=3, shrinking=True,
        tol=0.001, verbose=False)

    clf.fit(X_train, y_train)
    result= clf.predict(X_test)
    prob = clf.predict_proba(X_test)
    accuracy = getAccuracy(result,y_test)
    return accuracy

## Auto Encoder
def autoEncoder_(x_train,y_train,x_test,y_test):
    num_train = len(y_train)
    num_test = len(y_test)

    height, width, depth = 1, 37, 1 # MNIST images are 28x28
    num_classes = 7 # there are 10 classes (1 per digit)

    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = x_train
    y_train = y_train
    X_test = x_test
    y_test = y_test

    X_train = X_train.reshape(num_train, height * width)
    X_test = X_test.reshape(num_test, height * width)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # X_train /= 255 # Normalise data to [0, 1] range
    # X_test /= 255 # Normalise data to [0, 1] range

    Y_train = np_utils.to_categorical(y_train, num_classes) # One-hot encode the labels
    Y_test = np_utils.to_categorical(y_test, num_classes) # One-hot encode the labels

    input_img = Input(shape=(height * width,))

    x = Dense(height * width, activation='relu')(input_img)

    encoded = Dense(height * width, activation='relu')(x)
    encoded = Dense(height * width, activation='relu')(encoded)

    y = Dense(height * width, activation='relu')(x)

    decoded = Dense(height * width, activation='relu')(y)
    decoded = Dense(height * width, activation='relu')(decoded)

    z = Dense(height * width, activation='sigmoid')(decoded)
    model = Model(input_img, z)

    model.compile(optimizer='adadelta', loss='mse') # reporting the accuracy

    model.fit(X_train, X_train,
        epochs=100,
        batch_size=128,
        shuffle=True,
        validation_data=(X_test, X_test))

    mid = Model(input_img, y)
    reduced_representation =mid.predict(X_test)

    out = Dense(num_classes, activation='softmax')(y)
    reduced = Model(input_img, out)
    reduced.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

    reduced.fit(X_train, Y_train,
        epochs=100,
        batch_size=128,
        shuffle=True,
        validation_data=(X_test, Y_test))

    scores = reduced.evaluate(X_test, Y_test, verbose=1)
    print("scores",scores[0])
    return scores[1]*100

## Ensemble
def Ensemble_(train_data,train_label,test_data,test_label):
    clf = GradientBoostingClassifier()
    clf = clf.fit(train_data,train_label)
    result = clf.predict(test_data)
    accu = getAccuracy(result,test_label)
    return accu

## CNN
def cnn_(train_data,train_label,test_data,test_label):
    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
            beta_1=0.9, beta_2=0.999, early_stopping=False,
        epsilon=1e-08, hidden_layer_sizes=(150,), learning_rate='constant',
        learning_rate_init=0.001, max_iter=300, momentum=0.9,
        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
        solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
        warm_start=False)
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1)
    clf = clf.fit(train_data,train_label)
    result = clf.predict(test_data)
    accu = getAccuracy(result,test_label)
    return accu

## Ouput the result:
#svm_acc = svm_(x_train,y_train,x_test,y_test)
# random_forest_acc = RandomForest_(x_train,y_train,x_test,y_test)
# knn_accu = knn_(x_train,y_train,x_test,y_test)
# decison_tree_accu = decision_tree(x_train,y_train,x_test,y_test)
# ensemble_accu = Ensemble_(x_train,y_train,x_test,y_test)
# cnn_accu = cnn_(x_train,y_train,x_test,y_test)
auto_encoder_accu = autoEncoder_(x_train,y_train,x_test,y_test)

# print("KNN Accuracy ", knn_accu)
# print("SVM Accuracy",svm_acc)
# print("Random Forest Accuracy ",random_forest_acc)
# print("Decision Tree Accuracy ", decison_tree_accu)
print("Auto Encoder Accuracy ", auto_encoder_accu)
# print("Ensemble Accuracy ", ensemble_accu)
# print("CNN Accuracy",cnn_accu)