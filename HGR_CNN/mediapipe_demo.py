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

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    kc = KeyPointClassifier()

    with open('MediaPipeline/model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    conf = c.Configuration(version_name = "autoencoder", debug_mode=False, latest_model_name="prekazky_blured_and_prekazky_34k.h5")
    cap = vc.VideoImageCatcher(conf)
    cap.init_stream()

    with mp_hands.Hands(
        model_complexity = 0,
        max_num_hands = 2,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as hands:
        while True:
            image,depth = cap.get_depth_raw_color()
            dpt = cap.get_depth_img()
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            

            if results.multi_hand_landmarks:
                debug_image = copy.deepcopy(image)

                landmark_point = []

                # Keypoint
                for landmarks in results.multi_hand_landmarks:
                    landmark_x = [min(int(landmark.x * image.shape[1]), image.shape[1] - 1) for landmark in landmarks.landmark]
                    landmark_y = [min(int(landmark.y * image.shape[0]), image.shape[0] - 1) for landmark in landmarks.landmark]
                    # landmark_z = landmark.z

                    landmark_point = np.column_stack((landmark_x, landmark_y))

                pre_processed_landmark_list = pre_process_landmark(
                    landmark_point)

                hand_sign_id = kc(pre_processed_landmark_list)

                info_text = keypoint_classifier_labels[hand_sign_id]
                cv2.putText(image, info_text, (20,20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            #for hand_landmarks in results.multi_hand_landmarks:
            #    mp_drawing.draw_landmarks(
            #        image,
            #        hand_landmarks,
            #        mp_hands.HAND_CONNECTIONS,
            #        mp_drawing_styles.get_default_hand_landmarks_style(),
            #        mp_drawing_styles.get_default_hand_connections_style())

                rows, cols, _ = image.shape
                for landmarks in results.multi_hand_landmarks:
                    x = [landmark.x * cols for landmark in landmarks.landmark]
                    y = [landmark.y * rows for landmark in landmarks.landmark]
                    hand_coordinates = np.column_stack((x, y))
                cv2.circle(image,(int(hand_coordinates[8][0]),int(hand_coordinates[8][1])),10,(255,255,255),-1)

                hand_coordinates[8][0] = np.clip(hand_coordinates[8][0],0,cols-1)
                hand_coordinates[8][1] = np.clip(hand_coordinates[8][1],0,rows-1)
                #print(hand_coordinates[8])
                u = rs.rs2_deproject_pixel_to_point(cap.intrinsics,[int(hand_coordinates[8][0]),int(hand_coordinates[8][1])],depth[int(hand_coordinates[8][1]),int(hand_coordinates[8][0])]*cap.depth_scale)
                #a = u[0]
                #u[0] = u[1]
                #u[1] = -a
                #print(u)
                v=rs.rs2_transform_point_to_point(extr,u)
                #v = np.divide(v,1000) 
                #print(v)
                v.append(hand_sign_id)
            
                buf = struct.pack('%sf' % len(v), *v)
                s.sendto(buf,addr)
        
            cv2.imshow('',image)
            cv2.imshow('a',dpt)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.close_stream()
    s.detach()
    s.close()

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

if __name__ == '__main__':
    main()

