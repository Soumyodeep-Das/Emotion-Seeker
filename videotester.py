import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# load model
model = load_model('best_model.h5')

face_haar_cascade = cv2.CascadeClassifier( cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# capture video
cap=cv2.VideoCapture(0)

while True:
    ret, test_img=cap.read() # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

#     faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        # roi_gray=gray_img[y:y+w,x:x+h] # cropping region of interest i.e. face area from  image

        # Resize the grayscale image to (224, 224)
        resized_img = cv2.resize(gray_img[y:y + h, x:x + w], (224, 224))

        # Convert grayscale image to pseudo-RGB (3 channels)
        resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)

        # Normalize pixel values
        img_pixels = np.expand_dims(resized_img_rgb, axis=0)
        img_pixels = img_pixels / 255.0  # Normalize pixel values


#         roi_gray=cv2.resize(roi_gray,(224,224))
#         img_pixels = image.img_to_array(roi_gray)
#         img_pixels = np.expand_dims(img_pixels, axis = 0)
#         img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial Emotion Analysis', resized_img)

    if cv2.waitKey(10) == ord('q'): #wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows