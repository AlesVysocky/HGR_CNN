from pickletools import uint8
from PIL import Image
from pathlib import Path
from joblib import Parallel,delayed
import numpy as np
import os
from matplotlib import pyplot as plt


current_dir_path = os.path.dirname(os.path.realpath(__file__))

mask_dir_path = os.path.join(current_dir_path, "images")

mask2_dir_path = os.path.join(current_dir_path, "images-new")

def transform(filename):
    name = os.path.join(mask_dir_path,filename)
    newname = os.path.join(mask2_dir_path,filename)
    img = np.array(Image.open(name))
    out = img.astype(np.float32)
    out /= 10000.0
    out[out > 1.0] = 1.0
    out = out*255
    out[out>0]=255-out[out>0]
    im = Image.fromarray(np.uint8(out),mode="L")
    im.save(newname)

results = Parallel(n_jobs=4)(delayed(transform)(filename) for filename in os.listdir(mask_dir_path))
 

 

