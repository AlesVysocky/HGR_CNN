import numpy as np
import os
import cv2
import re


def get_img_name(img_counter, tip_pos, is_hand_detected, gesture):
    if not is_hand_detected: 
        tip_pos = (0,0,0)
        gesture = datatypes.Gesture.UNDEFINED
    timestamp = datetime.now().strftime("%m-%d-%Y_%H#%M#%S")
    is_hand_detected_binary = int(is_hand_detected * 1)
    # TODO put counter of image to the end of name
    return "rgbd_{}_X{}_Y{}_Z{}_hand{}_gest{}_date{}.png".format(img_counter, tip_pos[0], tip_pos[1], tip_pos[2], is_hand_detected_binary, gesture.value, timestamp)

class ImageDataManager:
    def __init__(self, main_script_path, dataset_dir, image_state_base, image_target_size, xyz_ranges):
        self.image_state_base = image_state_base
        self.main_script_path = main_script_path
        self.dataset_dir = dataset_dir
        self.image_target_size = image_target_size
        self.xyz_ranges = xyz_ranges

    def load_single_img(self, img_relative_path, ):
        img_path = os.path.join(self.main_script_path, img_relative_path)
        print("Loading image from %s ..." % img_path)
        X_img_data = self.__load_resized(img_path)
        y_expected = self.parse_expected_value(img_relative_path)
        print("Loaded: " + img_relative_path + " -> " + str(y_expected))
        return X_img_data, y_expected

    def get_train_data(self):
        X_train = []
        y_train = []
        for train_img_path in self.__get_train_images_names_from_folder():
            X_img_data, y_expected = self.load_single_img(os.path.join(self.dataset_dir, train_img_path))
            X_train.append(X_img_data)
            y_train.append(y_expected)
        return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

    def __get_train_images_names_from_folder(self):
        return list(filter(lambda x: x.startswith(self.image_state_base) and len(x) > 15, os.listdir(self.dataset_dir)))

    def parse_expected_value(self, img_name):
        result = []
        # TODO counter of image should be in the end of name
        float_val_group = "(-?\d+(?:.\d+)?)"
        regex_name_match = re.search('.*' + self.image_state_base + f'_\d+_X{float_val_group}_Y{float_val_group}_Z{float_val_group}_hand(.+)_gest(\d+)_date', img_name)
        for i in range(0,5):
            y_value = float(regex_name_match.group(i + 1))
            result.append(y_value)
        for i in range(0,3):  
            result[i] =  (result[i] + abs(self.xyz_ranges[i][0]))  / (abs(self.xyz_ranges[i][0]) +  self.xyz_ranges[i][1])
       
        return result

    def __load_resized(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #resized = cv2.resize(img, self.image_target_size)[:,:,3].astype(np.float32)  # TODO check if cast is required
        img = cv2.rotate(img, 0)
        img = cv2.resize(img,self.image_target_size).astype(np.float32)
        #cv2.imwrite("test-rot.png", img)

        return img[..., np.newaxis]


# TODO unit test
if __name__ == "__main__":
    #test_instance = ImageDataManager("", "", "rgbd", (1,1), xyz_ranges)
    #result = test_instance.parse_expected_value("rgbd_1_X-100.2_Y-200.12_Z300.12_hand1_gest2_date02-12-2020_10#34#14")
    #expected_result = [-100.2, -200.12, 300.12, 1, 2]
    #assert result == expected_result

    xyz_ranges = [(-700, 700), (-600, 500), (0, 1000)]
    test_instance = ImageDataManager("", "", "rgbd", (1,1),  [(-700, 700), (-600, 500), (0, 1000)])
    result = test_instance.parse_expected_value("rgbd_1_X0_Y-50_Z500_hand1_gest2_date02-12-2020_10#34#14")
    expected_result = [0.5, 0.5, 0.5, 1, 2]
    assert result == expected_result