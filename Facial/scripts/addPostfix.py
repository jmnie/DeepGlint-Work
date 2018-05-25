
import os
from shutil import copyfile


des_path = "F:\AffectNet\Manually_Annotated_aligned"
path_ = "F:\AffectNet\\result\manuallyannoimg"

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

file_path(path_,des_path)
