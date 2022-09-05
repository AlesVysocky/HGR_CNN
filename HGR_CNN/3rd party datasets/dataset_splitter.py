import os
import random
import shutil

current_dir_path = os.path.dirname(os.path.realpath(__file__))
image_dir_path = os.path.join(current_dir_path,"depth")
val_image_dir_path = os.path.join(current_dir_path,"depth-valid")
masks_dir_path = os.path.join(current_dir_path,"mask-hand")
val_masks_dir_path = os.path.join(current_dir_path,"mask-hand-valid")

l = os.listdir(masks_dir_path)
random.shuffle(l)

validation_subset_len = int(len(l)/5)

for i in range(validation_subset_len):
    print(f"Processing {i} from {validation_subset_len}")
    shutil.move(os.path.join(image_dir_path,l[i]),os.path.join(val_image_dir_path,l[i]))
    shutil.move(os.path.join(masks_dir_path,l[i]),os.path.join(val_masks_dir_path,l[i]))
