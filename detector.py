import os
import numpy as np
import cv2
import dlib
import sys

print(dlib.__version__)


face_file_path = sys.argv[1]
out_path = face_file_path +'/out'
os.mkdir(out_path)
predictor_path = './models/shape_predictor_5_face_landmarks.dat'

        
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

img_path = os.listdir(face_file_path)[0]
bgr_img = cv2.imread(face_file_path+img_path)
imgcp = np.copy(bgr_img)

if bgr_img is None:
   print("Sorry, we could not load '{}' as an image".format(face_file_path))
           
img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
dets = detector(img, 1)

num_faces = len(dets)
if num_faces != 0:
   print(" faces found in '{}'".format(face_file_path))
   
   faces = dlib.full_object_detections()
   for k,d in enumerate(dets):

       cv2.rectangle(imgcp,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,255),3)
       faces.append(sp(img, d))

       image = dlib.get_face_chip(img, faces[k],size=320)
       cv_rgb_image = np.array(image).astype(np.uint8)

       cv_bgr_img = cv2.cvtColor(cv_rgb_image, cv2.COLOR_RGB2BGR)
       des_path = out_path +'/'+ str(k) +".jpg"
       cv2.imwrite(des_path,cv_bgr_img)

   cv2.imwrite(out_path+"/masked.jpg",imgcp)


