import csv
import os
from shutil import copyfile
import json

train_path = "C:\\Users\Datasets\AffectNet\Manually_Annotated_Images"

LABELS = ["Neutral","Happiness","Sadness","Suprise","Fear","Disgust","Anger","Contempt","None","Uncertain","No-Face"]

## Read CSV

#train_csv = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\training.csv"
class dict_object:
    def __init__(self,co_1,co_2,co_3,co_4,co_5,co_6,co_7):
        self.co_1 = co_1
        self.co_2 = co_2
        self.co_3 = co_3
        self.co_4 = co_4
        self.co_5 = co_5
        self.co_6 = co_6
        self.co_7 = co_7

    def printself(self):
        print(self.co_1,self.co_2,self.co_3,self.co_4,self.co_5,self.co_6)

    def getLabel(self):
        return self.co_6

    def getFileName(self):
        return self.co_7


def read_csv(train_path):
    #train_path = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\training.csv"
    file_object = open(train_path)

    #file_dict = dict()

    csvReader = csv.reader(file_object)
    header = next(csvReader)

    filename = header.index("subDirectory_filePath")
    face_x = header.index("face_x")
    face_y = header.index("face_y")
    face_width = header.index("face_width")
    face_height = header.index("face_height")
    facial_landmarks = header.index("facial_landmarks")
    expression = header.index("expression")
    valence = header.index("valence")
    arousal = header.index("arousal")
    useless_label = [7,8,9,10]

    def getLandMark(landmarks):

        new_coordinate = []
        coordinate = landmarks.split(";")
        #print(len(coordinate),coordinate)

        length = int(len(coordinate)/2)

        for i in range(length):
            if coordinate[i*2+1] != '':
                #print("Right Length :",len(coordinate))
                new_coordinate.append([float(coordinate[i*2]),float(coordinate[i*2 + 1])])
            else:
                #print("Incorrect Length: ",len(coordinate))
                new_coordinate.append([float(coordinate[i*2]),0.0])

        return new_coordinate

    def getNewDir(text):
        search = '/'
        start = 0
        index = text.find(search,start)
        #print("Dir: ",str(text)[:index])
        return int(str(text)[:index])

    def getFileName(text):
        search = '/'
        start = 0
        index = text.find(search,start) + 1
        #print("index: ",index)
        return str(text)[index:]

    def findDot(text):
        search = '.'
        start = 0
        index = text.find(search, start)
        # print("index: ",index)
        return str(text)[:index]

    i = 0
    file_dict = dict()

    for row in csvReader:
        #filedir = getNewDir(row[filename])
        filedir = str(row[filename])
        _x = int(row[face_x])
        _y = int(row[face_y])
        _width = int(row[face_width])
        _height = int(row[face_height])
        landmark = getLandMark(row[facial_landmarks])
        label = int(row[expression])
        val = float(row[valence])
        aro = float(row[arousal])


        #print("File Dir ",filedir)
        # Remove the dirname
        #file_name = getFileName(row[filename])
        #print(file_name)
        # Remove postfix
        #file_name = findDot(file_name)

        #print(file_name)

        #temp_value = [_x,_y,_width,_height,landmark,label,filedir]
        temp_value = [label,LABELS[label],[val,aro]]
        file_dict[filedir] = temp_value
        #print("i :",i,filedir)

    #print("Iteration :",i)
    return file_dict

## Read File Path

def file_path(path):

    print(path)
    main_dir = os.listdir(path)
    file_path_dict = dict()

    for subdir in main_dir:

        filepath = os.path.join(path,subdir)

        sub_dirs = os.listdir(filepath)

        for subsubdir in sub_dirs:
            sub_filepath = os.path.join(filepath,subsubdir)
            #print(sub_filepath,subsubdir)
            #print(subdir,subsubdir) #subsubdir is the name of the file
            file_path_dict[subsubdir] = [subdir,sub_filepath]

    return file_path_dict

def copy_file(destination_path,filepath,filename):
    #destination_path = "C:\\Users\Datasets\AffectNet\images"
    path = "C:\\Users\Datasets\AffectNet\Manually_Annotated_Images"
    path_ = path + filepath
    target_path = destination_path + str('\\') + filename
    copyfile(path_, target_path)


    # print(path)
    # main_dir = os.listdir(path)
    # for subdir in main_dir:
    #     dirpath = os.path.join(path,subdir)
    #     sub_dirs = os.listdir(dirpath)
    #
    #     for file in sub_dirs:
    #         file_path = os.path.join(dirpath,file)
    #         #print(file_path)
    #         target_path = os.path.join(destination_path,file)
    #         #print(target_path)
    #         copyfile(file_path, target_path)


#file_path(train_path)
#copy_file()

## Train file dict:
#train_path = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\training.csv"
#vali_path = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\validation.csv"
#train_file_dict = read_csv(train_path)
#vali_file_dict = read_csv(vali_path)

#print("Train Useful (68 Landmarks) + Not 6 Basic Emotions:",len(train_file_dict.keys()))
#print("Vali Useful (68 Landmarks) + Not 6 Basic Emotions :",len(vali_file_dict.keys()))
#print("Total :",len(train_file_dict.keys()) + len(vali_file_dict.keys()))

#dataset_path = "C:\\Users\Datasets\AffectNet\Manually_Annotated_Images"
#file_path(dataset_path)

if __name__ == '__main__':
    train_path = '/media/jiaming/Seagate Backup Plus Drive/AffectNet/Manually_Annotated_file_lists/training.csv'
    vali_path = '/media/jiaming/Seagate Backup Plus Drive/AffectNet/Manually_Annotated_file_lists/validation.csv'

    train_dict = read_csv(train_path)
    vali_dict = read_csv(vali_path)

    with open('/media/jiaming/Seagate Backup Plus Drive/AffectNet/ini_mapping/train.json','w') as fp:
        json.dump(train_dict,fp)
    
    with open('/media/jiaming/Seagate Backup Plus Drive/AffectNet/ini_mapping/vali.json','w') as fp:
        json.dump(vali_dict,fp)
        
    #combine_dict = {**train_dict,**vali_dict}

    #with open('affect_dict.json','w') as fp:
    #    json.dump(combine_dict,fp)
    
    