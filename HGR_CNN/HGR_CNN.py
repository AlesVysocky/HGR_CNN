import cv2
import os
import sys
import tensorflow as tf
import pyrealsense2 as rs
import numpy as np
import datatypes
import image_data_manager as idm
import dataset_generator as gen
import cnn_model 
import predictor as pr
import SimulationPredictor as spredictor
from time import time, strftime, gmtime
import time as tm
from datetime import datetime
import video_fetcher as vf
import simulation_fetcher as sf
import visualizer as vis

version_name = "cnn-regression"

record_command = "record"
train_command = "train"
predict_command = "predict"
continue_train = "continue_train"
online_command = "online_prection"
simulation_command = "simulation_prection"

model_name = "current_model.h5"

filters_count = 32
learning_rate = 0.0005 # use high values because we have BatchNorm and Dropout
batch_size = 32
epochs_count = 20
test_data_ratio = 0.2

current_script_path = os.path.dirname(os.path.realpath(__file__))
date_time = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")
logs_dir = os.path.join(current_script_path, "logs", version_name+f"_LR{learning_rate}_{date_time}")
dataset_dir = os.path.join(current_script_path, os.pardir, "dataset") # pardir - parent dir (one lvl up)
models_dir = os.path.join(current_script_path, "models") 

record_when_no_hand = False
recorded_gesture = datatypes.Gesture.POINTING

img_camera_size = (640, 480) 
img_dataset_size = (320, 240)
camera_rate = 30

xyz_ranges = [(-700, 700), (-600, 500), (0, 1000)]
depth_max = float(1000) # millimeters
depth_min = float(0)
x_min = float(-700)
x_max = float(700)
y_min = float(-600)
y_max = float(500)


if __name__ == "__main__":

    #sys.argv = [sys.argv[0], record_command]
    #sys.argv = [sys.argv[0], train_command] 
    #sys.argv = [sys.argv[0], continue_train] 
    #sys.argv = [sys.argv[0], predict_command, os.path.join(dataset_dir, "depth_77_X356.4_Y-342.1_Z414.0_hand1_gest1_date03-02-2020_15-33-02.png")]
    #sys.argv = [sys.argv[0], predict_command, "depth_77_X356.4_Y-342.1_Z414.0_hand1_gest1_date03-02-2020_15-33-02.jpg"]
    #sys.argv = [sys.argv[0], online_command]
    sys.argv = [sys.argv[0], simulation_command]
    print(sys.argv) 

    image_manager = idm.ImageDataManager(current_script_path, dataset_dir, "depth", img_dataset_size, xyz_ranges) 
    visualizer = vis.Visualizer()
    
    if (len(sys.argv) == 1):
        print("No arguments provided. See help (-h).")
        sys.exit(0)

    if (sys.argv[1] == record_command):
        print("Dataset recording...")
        recorder = gen.DatasetGenerator(record_when_no_hand, dataset_dir, img_camera_size, img_dataset_size, xyz_ranges)
        recorder.record_data(recorded_gesture)
        sys.exit(0)

    if (sys.argv[1] == "tb"): #start tensorboard
        print("Starting tensorboard...")
        os.system('tensorboard --logdir='+os.path.join(current_script_path, "logs"))
        sys.exit(0)

    if (sys.argv[1] == train_command):
        print("Training...")
        model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, None)
        X_data, y_data = img_loader.get_train_data()
        model.train(X_data, y_data, epochs_count, batch_size, logs_dir, test_data_ratio)
        model.save(os.path.join(models_dir, "new_model.h5"))
        sys.exit(0)

    if (sys.argv[1] == continue_train): # FIXME  increases error!!!  probably because learning rate should be lower when cont.learning
        prev_model_name = "new_model.h5" # input from params
        print(f"Continue training of {prev_model_name}")
        model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, os.path.join(models_dir, prev_model_name))
        X_data, y_data = img_loader.get_train_data()
        model.train(X_data, y_data, epochs_count, batch_size, logs_dir, test_data_ratio)
        model.save(os.path.join(models_dir, "new_model.h5"))
        sys.exit(0)

    if (sys.argv[1] == online_command):
        print("Online prediction...")
        model_name = os.path.join(models_dir,"autoencoder_model.h5")
        predictor = pr.Predictor(model_name)
        video_fetcher = vf.VideoImageFetcher(img_camera_size,camera_rate)
        video_fetcher.init_stream()
        key = 1

        while key != 27:
            source = image_manager.encode_camera_image(video_fetcher.get_depth_raw())
            predicted = predictor.predict(source)
            visualizer.display_video("predicted",image_manager.decode_predicted(predicted),1)
            visualizer.display_video("camera image",video_fetcher.get_color(),1)

        video_fetcher.close_stream()
        sys.exit(0)

    if (sys.argv[1] == predict_command and not(sys.argv[2].isspace())):
        print("Predicting: %s" % sys.argv[2])
        tf.config.list_physical_devices('GPU') #check for gpu
        tf.config.set_visible_devices([], 'GPU') #  faster for prediction (2x), or loading?
        model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, os.path.join(models_dir, model_name))
        X_predict, y_predict = img_loader.load_single_img(sys.argv[2])
        #cv2.imwrite(os.path.join(current_script_path, "rgbd_6791_X261_Y175_Z356_hand1_gest1_date02-12-2020_14#41#54 TTT.png"), X_predict1)
        #X_predict, y_predict = img_loader.load_single_img(os.path.join(current_script_path, "rgbd_6791_X261_Y175_Z356_hand1_gest1_date02-12-2020_14#41#54 TTT.png"))
        result = model.predict_single_image(X_predict, y_predict)
        
        print("Expected: %s - predicted: %s" % (y_predict, result))
        for i in range(0,3):  
            result[i] = result[i] *  (abs(xyz_ranges[i][0]) +  xyz_ranges[i][1]) - abs(xyz_ranges[i][0])
        result = np.round(result).astype("int")
        # TODO !!! One Hot encoding for geture type prediction
        print("[X:%s; Y:%s; Z:%s; Hand:%s; Gesture:%s;]" % (result[0],result[1],result[2], result[3] == 1, result[4]))
        sys.exit(0)

    if (sys.argv[1] == simulation_command):
        print("Simulation prediction...")
        model_name = os.path.join(models_dir,"autoencoder_model.h5")
        predictor = pr.Predictor(model_name)
        sim_fetcher = sf.SimulationFetcher()
        sim_fetcher.init_stream()
        key = 1

        while key != 27:
            source = image_manager.encode_sim_image(sim_fetcher.get_depth_img())
            predicted = predictor.predict(source)
            visualizer.display_video("predicted",image_manager.decode_predicted(predicted),1)
            #visualizer.display_video("camera image",video_fetcher.get_color(),1)

        sim_fetcher.close_stream()
        sys.exit(0)

#obsolate:

    #if (sys.argv[1] == online_command):
    #    print("Online prediction...")
    #    model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, os.path.join(models_dir, model_name))
    #    predict = predictor.OnlinePredictor(model, img_camera_size, img_dataset_size, xyz_ranges)
    #    predict.predict_online()
    #    sys.exit(0)

      #if (sys.argv[1] == simulation_command):
      #  print("Simulation prediction...")
      #  model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, os.path.join(models_dir, model_name))
      #  spredict = spredictor.SimulationPredictor(model, img_camera_size, img_dataset_size, depth_max,depth_min,x_min,x_max,y_min,y_max, xyz_ranges)
      #  spredict.predict_online()
      #  sys.exit(0)