import cv2
import os

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name = input("Enter your name: ")
user_id = input("Enter numeric ID (1,2,3): ")

cam = cv2.VideoCapture(0)
count = 0

os.makedirs(f"dataset/{name}", exist_ok=True)

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"dataset/{name}/{user_id}_{count}.jpg", face_img)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Capture Faces", frame)

    if cv2.waitKey(1) == 27 or count >= 20:
        break

cam.release()
cv2.destroyAllWindows()
