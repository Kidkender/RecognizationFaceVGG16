from PIL import Image
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

screen_width = 1280       
screen_height = 720

image_width = 224
image_height = 224

model = load_model('model_trained/model_VGG16.h5')

with open("model_trained/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {key:value for key,value in og_labels.items()}
    print(labels)

stream = cv2.VideoCapture(0)

while(True):
    (grabbed, frame) = stream.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(
        rgb, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces: 
        roi_rgb = rgb[y:y+h, x:x+w]

        color = (255, 0, 0) # in BGR
        stroke = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

        size = (image_width, image_height)
        resized_image = cv2.resize(roi_rgb, size)
        image_array = np.array(resized_image, "uint8")
        img = image_array.reshape(1,image_width,image_height,3) 
        img = img.astype('float32')
        img /= 255

        predicted_prob = model.predict(img)

        font = cv2.FONT_HERSHEY_SIMPLEX
        name = labels[predicted_prob[0].argmax()]
        color = (255, 0, 255)
        stroke = 2
        cv2.putText(frame, f'({name})', (x,y-8),
            font, 1, color, stroke, cv2.LINE_AA)

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):    
            break      

stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
