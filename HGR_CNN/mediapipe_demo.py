import cv2
import video_catcher as vc
import config as c
import numpy as np
import UDP_client as UDP
import rs_utils as rs
import mp_recognizer as mp


HOST = "127.0.0.1"
PORT = 4023
ROT = [1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,-1.0]
TRANS = [0.062,-0.37,0.94]

udp = UDP.UDPClient(HOST,PORT)

conf = c.Configuration(version_name = "autoencoder", debug_mode=False, latest_model_name="prekazky_blured_and_prekazky_34k.h5")
cap = vc.VideoImageCatcher(conf)
cap.init_stream()

utils = rs.RSUtils(cap.intrinsics,ROT,TRANS,cap.depth_scale)

rec = mp.MPRecognizer(0,1,0.5,0.5)
    
while True:
    image,depth = cap.get_depth_raw_color()
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    hand_coordinates = rec.recognize(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    rec.draw_point(image,8)
    cols,rows,_=image.shape
    if hand_coordinates:
        hand_coordinates[8][0] = np.clip(hand_coordinates[8][0],0,cols-1)
        hand_coordinates[8][1] = np.clip(hand_coordinates[8][1],0,rows-1)
        u = utils.transform_px_to_pt([int(hand_coordinates[8][0]),int(hand_coordinates[8][1])],depth[int(hand_coordinates[8][1]),int(hand_coordinates[8][0])])
            
        v = utils.transform_pt_to_base(u)
        v.append(2.0)
        udp.send_floats(v)
        print(v)    
    cv2.imshow('',image)
    if cv2.waitKey(5) & 0xFF == 27:
           break
cap.close_stream()
udp.close()

