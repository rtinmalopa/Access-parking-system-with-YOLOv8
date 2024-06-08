from ultralytics import YOLO
from utils import *
from db import db_contains

results = {}

# Load the YOLOv8 model
coco_model = YOLO('models/yolov8n.pt')
license_plate_detector = YOLO('./models/30-epochs-gpu-trained-lpr-model.pt')

# Open the video file
video_path = "./assets/videos/audiA5-front-4k60fps.mp4"
cap = cv2.VideoCapture(video_path)

# car, motorcycle, bus, truck
vehicles_to_track = [2, 3, 5, 7]

# read frames
frame_nmr = -1
# ret = True
while cap.isOpened():
    frame_nmr += 1
    success, frame = cap.read()
    if success:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model.track(frame, classes=vehicles_to_track, persist=True)[0]
        vehicles = []

        for detection in detections.boxes.data.cuda().tolist():
            car_x1, car_y1, car_x2, car_y2, track_id, conf, class_id = detection
            if int(class_id) in vehicles_to_track:
                vehicles.append([car_x1, car_y1, car_x2, car_y2, track_id, conf])

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, _ = license_plate

            # assign license plate to car
            car_x1, car_y1, car_x2, car_y2, car_id, conf = get_car(license_plate, vehicles)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                # cv2.imshow("crop", license_plate_crop)

                # removes noisy colors from license plate
                adjusted_license_plate_crop = adjust_license_plate(license_plate_crop)
                # cv2.imshow("adjusted", adjusted_license_plate_crop)

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(adjusted_license_plate_crop, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("gray", license_plate_crop_gray)

                license_plate_text, license_plate_text_score = get_license_plate_details(license_plate_crop_gray)

                if license_plate_text is not None:
                    if db_contains(license_plate_text) and license_plate_text_score >= 0.9:
                        json_data = prepare_json_data(license_plate_text, license_plate_crop)
                        print(license_plate_text, license_plate_text_score)

                        # send_post_request(json_data)

                    results[frame_nmr][car_id] = {'car': {'bbox': [car_x1, car_y1, car_x2, car_y2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

        cv2.imshow('Webcam', frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

write_csv(results, './result.csv')
