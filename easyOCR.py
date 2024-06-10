from ultralytics import YOLO
from utils import *

model = YOLO('./models/200-epochs-gpu-trained-lpr-model.pt')

# img = cv2.imread('./assets/images/audiA4.PNG')
img = cv2.imread('./assets/images/20240306123724_IMG_5602.JPG')
# results = model.predict('./assets/images/audiA4.PNG', save=False, imgsz=320, conf=0.5)
results = model.predict('./assets/images/20240306123724_IMG_5602.JPG', save=False, imgsz=320, conf=0.5)


for license_plate in results:
    # for license_plate in license_plates:
    x1, y1, x2, y2 = license_plate.boxes.xyxy[0].tolist()
    # crop license plate
    license_plate_crop = img[int(y1):int(y2), int(x1):int(x2)]
    # cv2.imshow('crop', license_plate_crop)

    compare_images_ssim('./assets/images/20240306123724_IMG_5602.JPG', license_plate_crop)

    license_plate_text, license_plate_text_score = process_license_plate(license_plate_crop)
    print(license_plate_text, license_plate_text_score)

# Wait for a key press and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
