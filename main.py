import cv2
import argparse
import supervision as sv
import numpy as np
import pyttsx3
import time

from ultralytics import YOLO

ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    # Initialize pyttsx3
    tts_engine = pyttsx3.init()
    tts_engine.setProperty("rate", 150)  # Set speech rate
    last_announced = {}  # Dictionary to store last announced time for each object

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        # Trigger zone detection
        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        # Announce detected objects with a 5-second interval
        current_time = time.time()
        current_objects = {
            model.model.names[class_id]: current_time
            for _, _, class_id, _
            in detections
        }

        for obj, timestamp in current_objects.items():
            if obj not in last_announced or (current_time - last_announced[obj]) > 10:

                tts_engine.say(f"Detected {obj}")
                tts_engine.runAndWait()
                last_announced[obj] = current_time

        cv2.imshow("YOLOv8", frame)

        if cv2.waitKey(30) == 27:  # Press 'Esc' key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
