import mediapipe as mp
import numpy as np

class MPRecognizer:
    def __init__(self,model_complexity,max_num_hands,min_detection_confidence,min_tracking_confidence):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(True,max_num_hands,model_complexity,min_detection_confidence,min_tracking_confidence)

    def recognize(self,image):
        self.result = self.hands.process(image)

        if self.result.multi_hand_landmarks:
            rows, cols, _ = image.shape
            hand_coordinates = []
            for landmarks in self.result.multi_hand_landmarks:
               for landmark in landmarks.landmark:
                   hand_coordinates.append(self.mp_drawing._normalized_to_pixel_coordinates(landmark.x,landmark.y,rows,cols))
  
        else:
            return None;
        return hand_coordinates

    def draw_results(self,image):
        if self.result.multi_hand_landmarks:
            for hand_landmarks in self.result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        else:
            print("No results!")

    def draw_point(self,image,point):
        if self.result.multi_hand_landmarks:
            for i in range(len(self.result.multi_hand_landmarks)):
                for j in range(len(self.result.multi_hand_landmarks[i].landmark)):
                        if (j != point):
                            self.result.multi_hand_landmarks[i].landmark[j].visibility = 0

            self.draw_results(image)
        else:
            print("No results!")
