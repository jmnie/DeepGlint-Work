import os 
from shutil import copyfile

def TrainTestValiSplit(mapping_path,data_path,des_path):
    x_train_path = mapping_path + '/x_train.json'
    y_train_path = mapping_path + '/y_train.txt'
    x_test_path = mapping_path + '/x_test.json'
    y_test_path = mapping_path + '/y_test.txt'
    x_vali_path = mapping_path + '/x_vali.json'
    y_vali_path = mapping_path + '/y_vali.txt'

    with open(x_train_path,'r') as fp:
        x_train = json.load(fp)
    
    with open(x_test_path,'r') as fp:
        x_test = json.load(fp)
    
    with open(x_vali_path,'r') as fp:
        x_vali = json.load(fp)

    def createDirectroy(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def getDir(text):
        search = '/'
        start = 0
        index = text.find(search,start)
        #print("Dir: ",str(text)[:index])
        return str(text)[:index]

    for i in range(len(x_train)):
        key = x_train[i]
        dir = des_path + 'train/'+ getDir(key)
        file_path = data_path + key
        new_path = des_path + 'train/' + key
        createDirectory(dir)
        copyfile(file_path,new_path)
    
    for i in range(len(x_test)):
        key = x_test[i]
        dir = des_path + 'test/' + getDir(key)
        file_path = data_path + key
        new_path = des_path + 'test/' + key
        createDirectory(dir)
        copyfile(file_path,new_path)
    
    for i in range(len(x_vali)):
        key = x_vali[i]
        dir = des_path + 'vali/' + getDir(key)
        file_path = data_path + key
        new_path = des_path + 'vali/' + key
        createDirectory(dir)
        copyfile(file_path,new_path)
        
    

    