import glob
import os

def moving_files():
    folder_list = glob.glob(r"data\training\*")
    destination = r"data\LCCV\ann"

    for folder in folder_list:
        files = glob.glob(folder+"\\*")

        for file in files:
            if file.split('.')[-1] == 'png' and file.split('.')[-2][-2:]=='fg':
                os.rename(file, destination + folder[-3:] + file.split("\\")[-1] )

def renaming():
    folder = glob.glob(r"data\LCCV\ann\*")
    
    for file in folder:
        name = file.split(".")[-2][:-2]
        os.rename(file, name+'rgb.png' )

def sampling():
    import random
    import shutil
    folder_list = glob.glob(r"data\LCCV\train\*")
    destination = r"data\LCCV\val"

    validation_files = random.sample(folder_list, int(len(folder_list)*0.2))
    
    for validation_file in validation_files:
        shutil.move(validation_file, destination)


