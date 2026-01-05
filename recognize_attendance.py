import cv2
import csv
from datetime import datetime
import os

# ----------------------------
# 1️⃣ Load face cascade & recognizer
# ----------------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Make sure your trained model file path is correct
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/model.yml")  # adjust path if needed

# ----------------------------
# 2️⃣ Map numeric IDs to real names
# ----------------------------
names = {
    1: "Tharini"
   
}


# ----------------------------
# 3️⃣ Prepare camera & attendance tracking
# ----------------------------
cam = cv2.VideoCapture(0)
marked = set()  # to avoid duplicate entries
csv_file = "attendance.csv"

# Create CSV if not exists
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

# ----------------------------
# 4️⃣ Main loop: detect and recognize
# ----------------------------
while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_img)

        # Confidence < 80 means recognized
        if confidence < 150:
            name = names.get(label, "Tharini")
            


            # Log attendance only once
            if name not in marked:
                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, datetime.now().strftime("%H:%M:%S")])
                marked.add(name)

            cv2.putText(frame, f"{name}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Show live webcam
    cv2.imshow("Attendance System", frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

# ----------------------------
# 5️⃣ Release resources
# ----------------------------
cam.release()
cv2.destroyAllWindows()
print("Attendance session ended.")
