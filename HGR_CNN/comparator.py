import cv2
import os
import numpy as np


class Comparator:
   depth_dir = "Depth"
   label_dir = "Labeled"

   def __init__(self, model, config, image_manager,dataset):
        self.dataset = dataset
        self.output_dir = os.path.join(dataset,config.latest_model_name.split(".")[0])
        os.mkdir(self.output_dir)
        self.image_manager = image_manager
        self.config = config #not used
        self.model = model
        self.iteration = 0

   def compare_single_image(self,filename):
       depth_filename = os.path.join(self.dataset,self.depth_dir,filename)
       label_filename = os.path.join(self.dataset,self.label_dir,filename)
       label = cv2.imread(label_filename,0)
       original = cv2.imread(depth_filename,0)
       prediction = self.model.predict_single(original)
       prediction = cv2.cvtColor(prediction,cv2.COLOR_GRAY2BGR)
       FP = 0
       TP = 0
       FN = 0
       for i in range (prediction.shape[0]):
           for j in range (prediction.shape[1]):
                if prediction[i,j][0]>label[i,j]:
                   prediction[i,j]=(0,0,255)
                   FP = FP + 1
                elif prediction[i,j][0]<label[i,j]:
                   prediction[i,j]=(255,0,0)
                   FN = FN + 1
                elif prediction[i,j][0]==255:
                   TP = TP + 1
       output_filename = os.path.join(self.output_dir,filename)
       cv2.imwrite(output_filename,prediction)

       return [TP,FP,FN]
   def compare_dataset(self):
       log_filename = os.path.join(self.output_dir,"log.txt")
       stats =[]
       for name in os.listdir(os.path.join(self.dataset,self.depth_dir)):
           stats.append([name,*self.compare_single_image(name)])
       
       f = open(log_filename,"w")
       precision = []
       recall = []
       IoU = []
       for stat in stats:
           if (stat[1]+stat[2]==0):
               precision.append(-1)
           else:
               precision.append(stat[1]/(stat[1]+stat[2]))
           if (stat[1]+stat[3]==0):
               recall.append(-1)
           else:
               recall.append(stat[1]/(stat[1]+stat[3]))

           if (stat[1]+stat[3]+stat[2]>0):
            IoU.append(stat[1]/(stat[1]+stat[2]+stat[3]))
           else:
            IoU.append(-1)
           f.write("{} TP: {},FP: {}, FN: {}, P: {}, R: {}, IoU: {}\n".format(stat[0],stat[1],stat[2],stat[3],precision[-1],recall[-1],IoU[-1]))
           if (precision[-1]<0)or(recall[-1]<0)or(IoU[-1]<0):
            precision.pop()
            recall.pop()
            IoU.pop()

      
       f.write("mAP: {}, mAR: {}, mIoU: {}".format(np.mean(precision),np.mean(recall),np.mean(IoU)))
       f.close()
       





