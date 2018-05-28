from PIL import Image
import numpy as np
import os
import scipy.io
import json

def dataset_script():

    # im = Image.open('image.jpg')
    # im = (np.array(im))
    #
    # print(im.shape)
    # r = im[:, :, 0].flatten()
    # g = im[:, :, 1].flatten()
    # b = im[:, :, 2].flatten()
    # label = [1]

    with open('finalImage.json') as fp:
        data_dict = json.load(fp)

    def findDot(text):
        search = '.'
        start = 0
        index = text.find(search, start)
        # print("index: ",index)
        return str(text)[:index]

    label_ = []
    image_ = np.array([])
    raw_data_path = "F:\AffectNet\Processed\\final_data"
    subdirs = os.listdir(raw_data_path)

    i = 0
    for subdir in subdirs:
        sub_dirpath = os.path.join(raw_data_path,subdir)
        #print(sub_dirpath)

        files = os.listdir(sub_dirpath)
        for file in files:
            key_ = findDot(file)
            temp_label = data_dict[key_][0]
            label_.append(temp_label)

            file_path = os.path.join(sub_dirpath,file)

            im_ = Image.open('image.jpg')
            im_ = (np.array(im_))

            if i == 0:
                image_ = im_
            else:
                image_ = np.concatenate((image_,im_),axis= 0)

            i = i + 1

    scipy.io.savemat('images.mat',{'image':image_})

#dataset_script()
test_array = np.array([])
t2 = np.ones((2,2))

test_array = t2
test_array = np.concatenate((test_array,t2),axis = 0)
#test_array = np.concatenate((test_array,t2),axis = 0)
print(test_array)