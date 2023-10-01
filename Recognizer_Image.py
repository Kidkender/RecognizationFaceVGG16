from PIL import Image
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

image_width = 224
image_height = 224

model = load_model('model_trained/model_VGG16.h5')

with open("model_trained/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {key: value for key, value in og_labels.items()}

image_directory = "data_test"  

for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        path = os.path.join(image_directory, filename)
        frame = cv2.imread(path)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(
            rgb, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_rgb = rgb[y:y + h, x:x + w]

            size = (image_width, image_height)
            resized_image = cv2.resize(roi_rgb, size)
            image_array = np.array(resized_image, "uint8")
            img = image_array.reshape(1, image_width, image_height, 3)
            img = img.astype('float32')
            img /= 255

            predicted_prob = model.predict(img)

            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[predicted_prob[0].argmax()]
            color = (255, 0, 255)
            stroke = 2
            cv2.putText(frame, f'({name})', (x, y - 8),
                        font, 1, color, stroke, cv2.LINE_AA)
            scaled_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            cv2.imshow("Image", frame)
            cv2.waitKey(0)  
cv2.destroyAllWindows()
