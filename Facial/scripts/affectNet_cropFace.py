from affectNet_sort import read_csv
from affectNet_sort import file_path
from PIL import Image
import os

csv1_path = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\training.csv"
csv2_path = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\validation.csv"
dataset_path = "C:\\Users\Datasets\AffectNet\Manually_Annotated_Images"
crop_path = "F:\AffectNet\crop_image"

#file_1_dict = read_csv(csv1_path)
#file_2_dict = read_csv(csv2_path)
#fp_dict = file_path(dataset_path)

def cropImage(x1,y1,x2,y2,img_path,des_path):
    im = Image.open(img_path)
    box = (x1,y1,x2,y2)
    region = im.crop(box)
    region.save(des_path)

def createDirectroy(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def crop_script():
    file_1_dict = read_csv(csv1_path)
    file_2_dict = read_csv(csv2_path)

    print(len(file_1_dict.keys()))
    print(len(file_2_dict.keys()))

    for key in file_1_dict:
        x1 = file_1_dict[key][0]
        y1 = file_1_dict[key][1]
        x2 = file_1_dict[key][0] + file_1_dict[key][2]
        y2 = file_1_dict[key][1] + file_1_dict[key][3]
        dir = file_1_dict[key][-1]
        createDirectroy(crop_path + str('\\') + str(dir))
        img_path = dataset_path + str('\\') + str(dir) + str('\\')  + str(key)
        des_path = crop_path + str('\\') + str(dir) + str('\\')  + str(key)
        cropImage(x1,y1,x2,y2,img_path,des_path)
        #print("Key :",key)
        #print(x1,y1,x2,y2)
        #print(img_path)
        #print(des_path)

    for key in file_2_dict:
        x1 = file_2_dict[key][0]
        y1 = file_2_dict[key][1]
        x2 = file_2_dict[key][0] + file_2_dict[key][2]
        y2 = file_2_dict[key][1] + file_2_dict[key][3]
        dir = file_2_dict[key][-1]
        createDirectroy(crop_path + str('\\') + str(dir))
        img_path = dataset_path + str('\\') + str(dir) + str('\\')  + str(key)
        des_path = crop_path + str('\\') + str(dir) + str('\\')  + str(key)
        cropImage(x1,y1,x2,y2,img_path,des_path)
        #print("Key :",key)


crop_script()

# ##Test:
# img_path = "C:\\Users\Datasets\AffectNet\Manually_Annotated_Images\\689\\737db2483489148d783ef278f43f486c0a97e140fc4b6b61b84363ca.jpg"
# des_path = "F:\AffectNet\crop_image\\689\\737db2483489148d783ef278f43f486c0a97e140fc4b6b61b84363ca.jpg"
# createDirectroy(crop_path + str('\\') + str(689))
# cropImage(134,134,134+899,134+899,img_path,des_path)
