import csv
import os
from shutil import copyfile

train_path = "C:\\Users\Datasets\AffectNet\Manually_Annotated_Images"

LABELS = ["Neutral","Happiness","Sadness","Suprise","Fear","Disgust","Anger","Contempt","None","Uncertain","No-Face"]

## Read CSV

train_csv = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\training.csv"
class dict_object:
    def __init__(self,co_1,co_2,co_3,co_4,co_5,co_6):
        self.co_1 = co_1
        self.co_2 = co_2
        self.co_3 = co_3
        self.co_4 = co_4
        self.co_5 = co_5
        self.co_6 = co_6

    def printself(self):
        print(self.co_1,self.co_2,self.co_3,self.co_4,self.co_5,self.co_6)

    def getLabel(self):
        return self.co_6

def read_csv(train_path):
    #train_path = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\training.csv"
    file_object = open(train_path)

    file_dict = dict()

    csvReader = csv.reader(file_object)
    header = next(csvReader)

    filename = header.index("subDirectory_filePath")
    face_x = header.index("face_x")
    face_y = header.index("face_y")
    face_width = header.index("face_width")
    face_height = header.index("face_height")
    facial_landmarks = header.index("facial_landmarks")
    expression = header.index("expression")
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
        index = text.find(search,start) + 1
        #print("index: ",index)
        return str(text)[index:]

    for row in csvReader:
        filedir = getNewDir(row[filename])
        _x = int(row[face_x])
        _y = int(row[face_y])
        _width = int(row[face_width])
        _height = int(row[face_height])
        landmark = getLandMark(row[facial_landmarks])
        label = int(row[expression])

        if len(landmark) == 68:
            if label not in useless_label:
                temp_dict_object = dict_object(_x,_y,_width,_height,landmark,label)
                file_dict[filedir] = temp_dict_object

    return file_dict

## Read File Path

def file_path(path):

    print(path)
    main_dir = os.listdir(path)

    for subdir in main_dir:

        filepath = os.path.join(path,subdir)

        sub_dirs = os.listdir(filepath)

        for subsubdir in sub_dirs:
            sub_filepath = os.path.join(filepath,subsubdir)
            #print(sub_filepath,subsubdir)
            print(subsubdir) #subsubdir is the name of the file


def copy_file(destination_path):
    #destination_path = "C:\\Users\Datasets\AffectNet\images"
    path = "C:\\Users\Datasets\AffectNet\Manually_Annotated_Images"

    print(path)
    main_dir = os.listdir(path)
    for subdir in main_dir:
        dirpath = os.path.join(path,subdir)
        sub_dirs = os.listdir(dirpath)

        for file in sub_dirs:
            file_path = os.path.join(dirpath,file)
            #print(file_path)
            target_path = os.path.join(destination_path,file)
            #print(target_path)
            copyfile(file_path, target_path)


#file_path(train_path)
#copy_file()

## Train file dict:
train_path = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\training.csv"
vali_path = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\validation.csv"
train_file_dict = read_csv(train_path)
vali_file_dict = read_csv(vali_path)

print("Train Useful (68 Landmarks) + Not 6 Basic Emotions:",len(train_file_dict.keys()))
print("Vali Useful (68 Landmarks) + Not 6 Basic Emotions :",len(vali_file_dict.keys()))
print("Total :",len(train_file_dict.keys()) + len(vali_file_dict.keys()))

