import numpy as np
import os
import sys
import time
# dir_ = "/home/xjiangbh/ModelZoo_work/Data/flowers/"
dir_ = "/home/xjiangbh/ModelZoo_work/Data/ISIC2019/ISIC_2019_Training_Input/"
txt_file = 'train_ISIC2019.txt'
output = open(txt_file, 'w')
files = os.listdir(dir_)
for file in files:
    path_ = os.path.join(dir_, file)
    output.write(path_ + '\n')
output.close()