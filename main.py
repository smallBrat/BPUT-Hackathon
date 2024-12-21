import cv2
import time
import pyttsx3
from ultralytics import YOLO

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')  # Use the YOLOv8 model (replace with your model if needed)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust the speech rate

# Capture phone camera feed via IP (replace with your phone camera IP stream URL)
phone_camera_url = "http://192.168.1.101:8080/video"  # Example IP (update with actual)
phone_cam = cv2.VideoCapture(phone_camera_url)

# Check if the stream is accessible
if not phone_cam.isOpened():
    print("Error: Unable to access the phone camera stream.")
    exit()

# Timer for speech announcements
last_announcement_time = time.time()

# Process video feed
while True:
    # Read frame from the phone camera
    ret, frame = phone_cam.read()
    if not ret:
        print("Failed to read from the phone camera.")
        break

    # Resize frame for better performance and portrait display
    frame = cv2.resize(frame, (360, 640))  # Adjust size as needed

    # YOLOv8 predictions
    results = model.predict(source=frame, show=False)  # Phone camera

    # Get detected object names
    detected_objects = [result.names[class_id] for result in results for class_id in result.boxes.cls]

    # Make an announcement every 10 seconds
    current_time = time.time()
    if current_time - last_announcement_time > 10 and detected_objects:
        objects_to_announce = ", ".join(set(detected_objects))  # Avoid duplicate object names
        announcement = f"Detected objects are: {objects_to_announce}."
        print(announcement)
        engine.say(announcement)
        engine.runAndWait()
        last_announcement_time = current_time

    # Display results
    cv2.imshow("Phone Camera - YOLOv8", results[0].plot())

    # Exit loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
phone_cam.release()
cv2.destroyAllWindows()
