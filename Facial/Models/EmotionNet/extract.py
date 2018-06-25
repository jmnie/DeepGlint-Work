import csv
import os
import json 

def read_csv(train_path):

    LABELS = ["Neutral","Happiness","Sadness","Suprise","Fear","Disgust","Anger","Contempt","None","Uncertain","No-Face"]

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
        temp_value = [label,landmark]
        file_dict[filedir] = temp_value
        #print("i :",i,filedir)

    #print("Iteration :",i)
    return file_dict

if __name__ == "__main__":
    train_path = '/media/jiaming/Seagate Backup Plus Drive/AffectNet/Manually_Annotated_file_lists/training.csv'
    vali_path = '/media/jiaming/Seagate Backup Plus Drive/AffectNet/Manually_Annotated_file_lists/validation.csv'

    train_dict = read_csv(train_path)
    vali_dict = read_csv(vali_path)

    with open('train.json','w') as fp:
        json.dump(train_dict,fp)
    
    with open('vali.json','w') as fp:
        json.dump(vali_dict,fp)