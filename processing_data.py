import cv2
import os
import pickle
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

headshots_folder_name = "./dataset/duck"

image_width = 224
image_height = 224

facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

images_dir = os.path.join(".", headshots_folder_name)

current_id = 0
label_ids = {}

for root, _, files in os.walk(images_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
            path = os.path.join(root, file)

            label = os.path.basename(root).replace(" ", ".").lower()

        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1

        imgtest = cv2.imread(path, cv2.IMREAD_COLOR)
        image_array = np.array(imgtest, "uint8")

      
        faces =  facecascade.detectMultiScale(imgtest,
            scaleFactor=1.1, minNeighbors=5)

        if len(faces) != 1:
            print(f'---Photo skipped---\n')
            os.remove(path)
            continue

        for (x_, y_, w, h) in faces:

            face_detect = cv2.rectangle(imgtest,
                    (x_, y_),
                    (x_+w, y_+h),
                    (255, 0, 255), 2)
            plt.imshow(face_detect)
            plt.show()

            size = (image_width, image_height)

            roi = image_array[y_: y_ + h, x_: x_ + w]

            resized_image = cv2.resize(roi, size)
            image_array = np.array(resized_image, "uint8")

            os.remove(path)

            im = Image.fromarray(image_array)
            im.save(path)
