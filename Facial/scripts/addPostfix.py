
import os
from shutil import copyfile
import pickle
import numpy as np
import struct
import json
from PIL import Image

# des_path = "F:\AffectNet\Manually_Annotated_aligned"
# path_ = "F:\AffectNet\\result\manuallyannoimg"

def createDirectroy(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def file_path(path,des_path):

    print(path)
    main_dir = os.listdir(path)

    for subdir in main_dir:

        filepath = os.path.join(path,subdir)

        sub_dirs = os.listdir(filepath)

        for subsubdir in sub_dirs:

            sub_filepath = os.path.join(filepath,subsubdir)
            des_filepath = des_path + str('\\') + str(subdir) + str('\\')+ str(subsubdir) + ('.jpg')
            des_dir = des_path + str('\\') + str(subdir)

            createDirectroy(des_dir)
            #print(sub_filepath)
            #print(des_filepath)
            copyfile(sub_filepath,des_filepath)

#file_path(path_,des_path)

def addPostfix(path_):
    subdirs = os.listdir(path_)

    for subdir in subdirs:
        subdir_path = os.path.join(path_,subdir)
        files = os.listdir(subdir_path)

        for file in files:
            file_path = os.path.join(subdir_path,file)
            new_name = file_path + ".jpg"
            os.rename(file_path,new_name)

def make_dictionary():
    origin_path = "F:\AffectNet\Manually_Annotated_Images"
    subdirs = os.listdir(origin_path)
    postfix_dict = dict()

    def getExt(filename):
        name, ext = os.path.splitext(filename)
        return name, ext

    for subdir in subdirs:
        dir_path = os.path.join(origin_path,subdir)
        files = os.listdir(dir_path)
        for file in files:
            filepath = os.path.join(dir_path,file)
            name,ext = getExt(file)
            #print(name)
            #print(ext)
            temp_key = str(subdir) + '/' + str(name)
            #print(temp_key)
            postfix_dict[temp_key] = ext

    with open("postFixDict_2.json",'w') as fp:
        json.dump(postfix_dict,fp)

#make_dictionary()
def add_postfix(spath, despath):
    subdirs = os.listdir(spath)

    with open("postFixDict_2.json", 'r') as fp:
        postfix_dict = json.load(fp)

    print("Program started:")
    i = 0
    for subdir in subdirs:
        dir_path = os.path.join(spath, subdir)
        files = os.listdir(dir_path)


        for file in files:
            filepath = os.path.join(dir_path, file)
            key_ = subdir + '/' + str(file)

            if key_ in postfix_dict.keys():
                post_fix = postfix_dict[key_]

                file_des = despath + '/' + str(subdir) + '/' + str(file) + post_fix
                des_dir = despath + '/' + str(subdir)
                createDirectroy(des_dir)
                copyfile(filepath, file_des)
                i = i + 1

    print("Program ended. Total number: ", i)

def addPostFix_handleError(spath, despath, subdir):

    with open("postFixDict_2.json", 'r') as fp:
        postfix_dict = json.load(fp)

    files = os.listdir(spath)

    i = 0

    for file in files:
        source_path_file = os.path.join(spath,file)
        key_ = subdir + '/' + str(file)
        file_des_path = os.path.join(despath,file)

        if key_ in postfix_dict.keys():
            post_fix = postfix_dict[key_]
            file_des_path = file_des_path + post_fix
            createDirectroy(despath)
            copyfile(source_path_file,file_des_path)
            i = i + 1

    print("Program ended. Total number: ", i)

def final_crop(des_path, test_path):
    # des_path = "F:\AffectNet\Processed\\final_data_align"
    # test_path = "F:\\result\manuallyannoimg"

    error_number = 0
    error_dict = dict()
    for subdir in os.listdir(test_path):
        filepath = os.path.join(test_path, subdir)
        sub_dirs = os.listdir(filepath)

        for file in sub_dirs:
            final_path_ = os.path.join(filepath, file)

            des_file_path_ = os.path.join(des_path, subdir)
            des_file_path = os.path.join(des_file_path_, file)

            #error_number = 0

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
                print("OS Error. filepath: ", final_path_)
                error_number += 1
                error_dict[file] = str(subdir)

    return error_dict
    #with open("error_dict.json",'w') as fp:
    #    json.dump(error_dict,fp)
    #print("Error Number: ", error_number)

#spath = 'F:\\result\manuallyannoimg'
despath = 'F:\\AffectNet\Processed\Mannually_image_postfix'
final_path = 'F:\AffectNet\Processed\\224crop_1'
#make_dictionary()
#add_postfix(spath, despath)
#final_crop(final_path,despa# th)

def RemoveFile(dirpath):

    createDirectroy(dirpath)

    for file in os.listdir(dirpath):
        file_path = os.path.join(dirpath, file)
        os.remove(file_path)

def crop_error_(source_path,des_path,subdir):
    error_dict = dict()

    for file in os.listdir(source_path):
        file_source_path = os.path.join(source_path,file)
        file_des_path = os.path.join(des_path,file)

        try:
            img = Image.open(file_source_path)
            x = 113 - 14
            y = 130 - 30
            length = 224
            area = (x, y, x + length, y + length)
            cropped_img = img.crop(area)

            #createDirectroy(des_file_path_)
            cropped_img.save(file_des_path)

        except OSError:
            print("OS Error. filepath: ", file_source_path)
            error_dict[file] = str(subdir)

    return error_dict


def HandlerTheError():

    print("Program started. ")
    with open("error_dict.json",'r') as fp:
        error_dict = json.load(fp)

    values = list(set(error_dict.values()))

    source_path = "F:\\new\\result\manuallyannoimg"
    despath = 'F:\\AffectNet\Processed\Mannually_image_postfix'

    path_list = []

    for dir in values:
        temp_source = source_path + '\\' + dir
        temp_des = despath + "\\" + dir
        path_list.append([temp_source,temp_des,dir])

    for path_ in path_list:
        print(path_)
        #RemoveFile(path_[1])
        addPostFix_handleError(path_[0],path_[1],path_[2])

    print("Program Ended. ")

def Handle_Crop_Error():

    new_error_dict = dict()

    print("Program started. ")
    with open("error_dict.json", 'r') as fp:
        error_dict = json.load(fp)

    values = list(set(error_dict.values()))


    source_path = 'F:\\AffectNet\Processed\Mannually_image_postfix'
    despath = "F:\AffectNet\Processed\\224crop_1"

    path_list = []

    for dir in values:
        temp_source = source_path + '\\' + dir
        temp_des = despath + "\\" + dir
        path_list.append([temp_source, temp_des, dir])

    for path_ in path_list:
        print(path_)
        #RemoveFile(path_[1])

        sub_error_dict = crop_error_(path_[0],path_[1],path_[2])
        new_error_dict = {**error_dict,**sub_error_dict}

    with open("error_handle_crop.json",'w') as fp:
        json.dump(new_error_dict,fp)

    print("Program Ended. ")

#HandlerTheError()
#Handle_Crop_Error()

def statistics():
    with open("error_handle_crop.json",'r') as fp:
        error_dict = json.load(fp)

    with open("postFixDict_2.json",'r') as fp:
        all = json.load(fp)

    with open("dataset_key_error.json",'r') as fp:
        error_key = json.load(fp)

    #print("Length :",- len(error_dict.keys()) + len(all.keys()))
    print("Length of ublabelled images :",len(error_key.keys()))

statistics()