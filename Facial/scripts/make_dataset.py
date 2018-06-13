from PIL import Image
import numpy as np
import os
import scipy.io
import json
from affectNet_sort import read_csv

def affectNet_dict():

    train_csv = "/media/jiaming/Seagate Backup Plus Drive/AffectNet/Manually_Annotated_file_lists/training.csv"
    vali_csv = "/media/jiaming/Seagate Backup Plus Drive/AffectNet/Manually_Annotated_file_lists/validation.csv"

    train_dict = read_csv(train_csv)
    vali_dict = read_csv(vali_csv)
    merge_dict = {**train_dict,**vali_dict}

    with open("AffectNet.json",'w') as fp:
        json.dump(merge_dict,fp)

    print("Done")

def train_vali_dict():
    train_csv = "/media/jiaming/Seagate Backup Plus Drive/AffectNet/Manually_Annotated_file_lists/training.csv"
    vali_csv = "/media/jiaming/Seagate Backup Plus Drive/AffectNet/Manually_Annotated_file_lists/validation.csv"

    train_dict = read_csv(train_csv)
    vali_dict = read_csv(vali_csv)
    #merge_dict = {**train_dict,**vali_dict}

    with open("/media/jiaming/Seagate Backup Plus Drive/AffectNet/train_test_json/train.json",'w') as fp:
        json.dump(train_dict,fp)
    
    with open("/media/jiaming/Seagate Backup Plus Drive/AffectNet/train_test_json/vali.json","w") as fp:
        json.dump(vali_dict,fp)

    print("Done")

def createDirectroy(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def dataset_script():

    # im = Image.open('image.jpg')
    # im = (np.array(im))
    #
    # print(im.shape)
    # r = im[:, :, 0].flatten()
    # g = im[:, :, 1].flatten()
    # b = im[:, :, 2].flatten()
    # label = [1]

    with open('AffectNet.json') as fp:
        label_dict = json.load(fp)

    key_error_dict = dict()

    label_ = []
    image_ = np.array([])
    raw_data_path = "F:\AffectNet\Processed\\224crop_1"
    subdirs = os.listdir(raw_data_path)

    i = 0
    for subdir in subdirs:
        sub_dirpath = os.path.join(raw_data_path,subdir)
        #print(sub_dirpath)

        files = os.listdir(sub_dirpath)

        for file in files:
            key_ = str(subdir) + '/' + str(file)
            file_path = os.path.join(sub_dirpath, file)

            if key_ in label_dict:
                temp_label = label_dict[key_][0]
                label_.append(temp_label)

                im_ = Image.open(file_path)
                im_ = (np.array(im_))

                if i == 0:
                    image_ = im_
                else:
                    image_ = np.concatenate((image_,im_),axis= 0)

                i = i + 1
            else:
                key_error_dict[key_] = file_path

    with open("dataset_key_error.json",'w') as fp:
        json.dump(key_error_dict,fp)

    label_ = np.array(label_)
    np.savetxt("C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\labels.txt",label_)
    scipy.io.savemat('C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\images.mat',{'image':image_})


def distributed_dataset():
    print("Distributed Dataset")

    with open('/train/execute/AffectNet/AffectNet.json') as fp:
        label_dict = json.load(fp)

    key_error_dict = dict()

    # label_ = []
    # image_ = np.array([])
    raw_data_path = "/train/execute/AffectNet/Data/finalData"
    subdirs = os.listdir(raw_data_path)
    despath = "/train/execute/AffectNet/Data/Dataset"
    iteration = 0
    for subdir in subdirs:
        sub_dirpath = os.path.join(raw_data_path, subdir)
        # print(sub_dirpath)

        files = os.listdir(sub_dirpath)

        sub_label = []
        sub_image = np.array([])

        print("Iteration :", iteration)
        i = 0

        for file in files:
            key_ = str(subdir) + '/' + str(file)
            file_path = os.path.join(sub_dirpath, file)

            if key_ in label_dict:
                temp_label = label_dict[key_][0]
                sub_label.append(temp_label)

                im_ = Image.open(file_path)
                im_ = (np.array(im_))

                if i == 0:
                    sub_image = im_
                else:
                    sub_image = np.concatenate((sub_image, im_), axis=0)

                i = i + 1
            else:
                key_error_dict[key_] = file_path

        des_dir = despath + str('/') + str(subdir)
        createDirectroy(des_dir)
        label_path = despath + str('/') + str(subdir) + '/' + str(subdir) + "label.txt"
        sub_img_path = despath + str('/') + str(subdir) + '/' + str(subdir) + "data.mat"

        np.savetxt(label_path,sub_label)
        scipy.io.savemat(sub_img_path,{'image':sub_image})
        iteration += 1

    print("Make Distributed Dataset Completed ")

# Test
# #dataset_script()
# test_array = np.array([])
# t2 = np.ones((2,2))
#
# test_array = t2
# test_array = np.concatenate((test_array,t2),axis = 0)
# #test_array = np.concatenate((test_array,t2),axis = 0)
# print(test_array)
#dataset_script()
#affectNet_dict()
#dataset_script()

if __name__ == "__main__":
    train_vali_dict()
