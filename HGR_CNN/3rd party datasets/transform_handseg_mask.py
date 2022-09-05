import cv2
import imutils
from pathlib import Path
import random
import os
import shutil
import numpy as np
from joblib import Parallel,delayed

current_dir_path = os.path.dirname(os.path.realpath(__file__))

mask_dir_path = os.path.join(current_dir_path, "masks")

mask2_dir_path = os.path.join(current_dir_path, "masks-new")

def transform(filename):
    name = os.path.join(mask_dir_path,filename)
    newname = os.path.join(mask2_dir_path,filename)
 
    img = cv2.imread(name,0)
    ar = np.array(img)
    ar[ar>0] = 255
    cv2.imwrite(newname,ar)
    
found = False

results = Parallel(n_jobs=4)(delayed(transform)(filename) for filename in os.listdir(mask_dir_path))
 


 

