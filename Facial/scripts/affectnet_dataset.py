import json
import os
from affectNet_sort import read_csv
from PIL import Image
import numpy as np

path = "C:\\Users\Jiaming Nie\Documents\GitHub\DeepGlint-Work\Facial\scripts\\affectNet.json"
LABELS = ["Neutral","Happiness","Sadness","Suprise","Fear","Disgust","Anger","Contempt","None","Uncertain","No-Face"]

def getMergeDict():
    training_path = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\training.csv"
    test_path = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\validation.csv"
    train_dict = read_csv(training_path)
    test_dict = read_csv(test_path)
    merge_dict = {**train_dict,**test_dict}

    # with open('affectNet.json', 'w') as fp:
    #     json.dump(final_label, fp)
    return merge_dict

def findDot(text):
    search = '.'
    start = 0
    index = text.find(search,start)
    #print("index: ",index)
    return str(text)[:index]

def data_folder_path(path):
    print(path)
    main_dir = os.listdir(path)
    file_path_dict = dict()

    i = 0
    for subdir in main_dir:

        filepath = os.path.join(path, subdir)
        sub_dirs = os.listdir(filepath)

        for subsubdir in sub_dirs:
            sub_filepath = os.path.join(filepath, subsubdir)
            # print(sub_filepath,subsubdir)
            # print(subdir,subsubdir) #subsubdir is the name of the file
            #print(subsubdir)
            subsubdir = findDot(subsubdir)
            file_path_dict[subsubdir] = [subdir, sub_filepath]
            #print(subsubdir)
            i = i + 1
    print("Total files :",i)

    return file_path_dict

def getRemaining():
    folder_path = "F:\AffectNet\Manually_Annotated_aligned"
    file_path_dict = data_folder_path(folder_path)
    mergeDict = getMergeDict()

    label_non_use = [8,9,10]
    final_label = dict()
    not_exist= dict()

    for key in file_path_dict:

        j = 0
        if key in mergeDict:
            if mergeDict[key][5] not in label_non_use:
                label = mergeDict[key][5]
                final_label[key] = [label,LABELS[label]]
            else:
                not_exist[key] = file_path_dict[key]
        else:

            j = j + 1
            not_exist[key] = file_path_dict[key]
            #os.remove(file_path_dict[key][1])

    print("MergeDict length ",len(mergeDict.keys()))
    print("Remaining files",len(final_label.keys()))
    print("Remove Length ",len(not_exist))

    with open('finalImage.json','w') as fp:
        json.dump(final_label,fp)

    with open('notExist.json','w') as fp:
        json.dump(not_exist,fp)

def del_non_exist():
    with open('notExist.json') as fp:
        non_exist = json.load(fp)

    print("Length ",len(non_exist))
    for key in non_exist:

        try:
            os.remove(non_exist[key][1])
        except FileNotFoundError:
            print("Not exist,continue")
            continue

def crop_demo():#source_folder,target_folder):
    test_path = "F:\AffectNet\Processed\Manually_Annotated_aligned\\11"
    des_path = "F:\AffectNet\Processed\\test_folder\\crop_image"

    for (dirpath, dirnames, filenames) in os.walk(test_path):

       for file in filenames:
           sub_path = os.path.join(test_path, file)
           #print(file)

           img = Image.open(sub_path)
           x = 113 - 14
           y = 130 - 30
           length = 224
           area = (x, y, x+length, y+length)
           cropped_img = img.crop(area)

           new_path = os.path.join(des_path,file)
           cropped_img.save(new_path)

def createDirectroy(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def final_crop():
    des_path = "F:\AffectNet\Processed\\final_data_align"
    test_path = "F:\\result\manuallyannoimg"

    for subdir in os.listdir(test_path):
        filepath = os.path.join(test_path, subdir)
        sub_dirs = os.listdir(filepath)

        for file in sub_dirs:
            final_path_ = os.path.join(filepath,file)

            des_file_path_ = os.path.join(des_path,subdir)
            des_file_path = os.path.join(des_file_path_,file)

            if not os.path.exists(des_file_path):
                #os.makedirs(des_file_path_)
                #print(des_file_path)
                #print(des_file_path_)

                # Read Origin Image and crop
                try:
                    img = Image.open(final_path_)
                    x = 113 - 14
                    y = 130 - 30
                    length = 224
                    area = (x, y, x + length, y + length)
                    cropped_img = img.crop(area)

                    createDirectroy(des_file_path_)
                    cropped_img.save(des_file_path)

                except OSError:
                    print("OS Error. filepath: ",final_path_)
                    continue
            else:
                continue

        # for file in filenames:
        #     sub_path = os.path.join(test_path, file)
        #     # print(file)
        #
        #     img = Image.open(sub_path)
        #     x = 113 - 14
        #     y = 130 - 30
        #     length = 224
        #     area = (x, y, x + length, y + length)
        #     cropped_img = img.crop(area)
        #
        #     new_path = os.path.join(des_path, file)
        #     cropped_img.save(new_path)

    #cropped_img.show()
#del_non_exist()
# getRemaining()

#crop_demo()

#final_crop()

def test():
    img = Image.open('/home/jiaming/code/github/DeepGlint-Work/Facial/scripts/image.jpg')
    x = 112
    y = 112
    length = 160
    array = (np.array(img))
    #print(array.shape)
    area = (x-int(0.5*length), y-int(0.5*length), x+int(0.5*length), y+int(0.5*length))
    cropped_img = img.crop(area)
    cropped_img.show()
    #createDirectroy(des_file_path_)
    #cropped_img.save(des_file_path)


if __name__ == '__main__':
    #test()
    import math
    print(-math.log(0.39))