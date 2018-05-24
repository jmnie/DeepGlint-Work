
from affectNet_sort import read_csv, copy_file
import json

no_face_file_path = "F:\AffectNet\\result\\manuallyannoimg_noface.txt"
file = open(no_face_file_path)
line = file.readline()

LABELS = [
   "Neutral",
   "Happiness",
   "Sadness",
   "Surprise",
   "Fear",
   "Disgust",
   "Anger",
   "Contempt",
   "None",
   "Uncertain",
   "No-Face"
]

def getFileName(text):
    search = '/'
    start = 0
    index = text.find(search, start) + 1
    # print("Dir: ",str(text)[:index])
    length = len(text) - 1
    return str(text)[index:length]

index_list = []

while line:
    #print(line, end='')
    #line = line.strip('\n')
    line = getFileName(line)
    index_list.append(line)
    line = file.readline()

file.close()

print(len(index_list))

# training_path = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\training.csv"
# test_path = "C:\\Users\Jiaming Nie\Downloads\Manually_Annotated_file_lists\\validation.csv"
# train_dict = read_csv(training_path)
# test_dict = read_csv(test_path)
# merge_dict = {**train_dict,**test_dict}

# with open('affectNet.json') as fp:
#     merge_dict = json.load(fp)
# #
# result_dict = dict()
#
# singular_file = dict()
# for filename in index_list:
#
#     if filename in merge_dict:
#         label = merge_dict[filename][5]
#         result_dict[filename] = [label,LABELS[label]]
#     else:
#         singular_file[filename] = ['Singular File']
#
#
# with open('result.json', 'w') as fp:
#     json.dump(result_dict, fp)

#with open('singular_file.json','w') as fp:
#     json.dump(singular_file,fp)

# with open('singular_file.json') as fp:
#     sing_file = json.load(fp)

# with open('result.json') as fp:
#     result = json.load(fp)

# print("Singular file Length ",len(sing_file.keys()))
# print("Result file length ",len(result.keys()))

# basic_emotion_face_ = dict()
# non_basic = [8,10]
#
# for key in result:
#     if result[key][0] not in non_basic:
#         subdir = merge_dict[key][6]
#         label = result[key][0]
#         des_ = result[key][1]
#         basic_emotion_face_[key] = [label,des_,subdir]


des_foler = "C:\\Users\Datasets\\no_face"

with open('result_no_face.json') as fp:
    basic_emotion_face_ = json.load(fp)

print(len(basic_emotion_face_.keys()))

for key in basic_emotion_face_:
    sub_dir = basic_emotion_face_[key][2]
    fpath = str('\\') + str(sub_dir) + str('\\') + str(key)
    copy_file(des_foler, fpath, key)

# with open('result_no_face.json','w') as fp:
#     json.dump(basic_emotion_face_, fp)
