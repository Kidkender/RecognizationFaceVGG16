import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

screen_width = 1280  
screen_height = 720
name_User="Tuan"
stream = cv2.VideoCapture(0)

output_directory = f'dataset/{name_User}' 
os.makedirs(output_directory, exist_ok=True)

image_count = 0

while True:
    (grabbed, frame) = stream.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        color = (0, 255, 255)  
        stroke = 5
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

        face_image = frame[y:y + h, x:x + w]
        image_count += 1
        image_filename = os.path.join(output_directory, f'{name_User}_{image_count}.png')
        cv2.imwrite(image_filename, face_image)

        print(f'Saved {image_filename}')

    cv2.imshow("Image", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or image_count >= 500:
        break

stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
