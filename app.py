from flask import Flask, request, jsonify, render_template
import cv2
import csv
from datetime import datetime
import os

app = Flask(__name__)

# Make sure attendance.csv exists
if not os.path.exists("attendance.csv"):
    with open("attendance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Time"])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/capture", methods=["POST"])
def capture_attendance():
    name = request.form.get("name").strip()
    if not name:
        return jsonify({"status": "error", "message": "Name cannot be empty."})

    # Open webcam
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cam.release()

    if not ret:
        return jsonify({"status": "error", "message": "Failed to capture from camera."})

    # Save captured image (optional)
    cv2.imwrite(f"captures/{name}_{datetime.now().strftime('%H%M%S')}.jpg", frame)

    # Mark attendance in CSV
    with open("attendance.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, datetime.now().strftime("%H:%M:%S")])

    return jsonify({"status": "success", "name": name, "time": datetime.now().strftime("%H:%M:%S")})

@app.route("/get_attendance")
def get_attendance():
    data = []
    with open("attendance.csv") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            data.append({"name": row[0], "time": row[1]})
    return jsonify(data)

if __name__ == "__main__":
    # Create captures folder if it doesn't exist
    if not os.path.exists("captures"):
        os.makedirs("captures")
    app.run(debug=True)
