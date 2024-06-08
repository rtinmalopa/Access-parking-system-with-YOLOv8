import string
import easyocr
import cv2
import numpy as np
import base64
import json
import uuid
import requests
from datetime import datetime

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Mapping dictionaries for character conversion
map_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5', 'B': '8'}
map_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S', '8': 'B'}


def write_csv(results, output_path):
    """
    Write the results to a CSV file.

    Args:
        results (dict): Dictionary containing the results.
        output_path (str): Path to the output CSV file.
    """
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                print(results[frame_nmr][car_id])
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )
        f.close()


def license_complies_format(text):
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in map_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in map_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in map_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in map_char_to_int.keys()) and \
       (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in map_char_to_int.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in map_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in map_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    mapping = {
        0: map_int_to_char,
        1: map_int_to_char,
        2: map_char_to_int,
        3: map_char_to_int,
        4: map_char_to_int,
        5: map_int_to_char,
        6: map_int_to_char
    }

    license_plate = ''.join(mapping[j].get(text[j], text[j]) for j in range(7))

    return license_plate


def read_license_plate(license_plate_crop):
    custom_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    detections = reader.readtext(license_plate_crop, allowlist=custom_characters)

    all_texts = []
    all_scores = []

    for _, text, score in detections:
        text = text.upper().replace(' ', '')
        all_texts.append(text)
        all_scores.append(score)

    license_plate_text = ''.join(all_texts)
    license_plate_score = sum(all_scores) / len(all_scores) if all_scores else 0

    if license_complies_format(license_plate_text):
        return format_license(license_plate_text), license_plate_score

    return None, None


def get_car(license_plate, vehicles):
    x1, y1, x2, y2, score, class_id = license_plate

    for i in range(len(vehicles)):
        car_x1, car_y1, car_x2, car_y2, track_id, _ = vehicles[i]

        if x1 > car_x1 and y1 > car_y1 and x2 < car_x2 and y2 < car_y2:
            return vehicles[i]

    return -1, -1, -1, -1, -1, -1


def adjust_license_plate(license_plate_crop):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_dark_blue = np.array([100, 150, 0])
    upper_dark_blue = np.array([140, 255, 100])

    # Define range of red color in HSV
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    lower_dark_red = np.array([0, 100, 20])
    upper_dark_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only blue and red colors
    mask_blue1 = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_blue2 = cv2.inRange(hsv, lower_dark_blue, upper_dark_blue)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_dark_red = cv2.inRange(hsv, lower_dark_red, upper_dark_red)

    mask_blue = cv2.bitwise_or(mask_blue1, mask_blue2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red = cv2.bitwise_or(mask_red, mask_dark_red)

    # Combine the blue and red masks
    mask = cv2.bitwise_or(mask_blue, mask_red)

    # Invert the mask to get everything except blue and red
    mask_inv = cv2.bitwise_not(mask)

    # Create a white image of the same size as the frame
    white_background = np.full(license_plate_crop.shape, 255, dtype=np.uint8)

    # Use the inverted mask to replace the blue and red areas with white
    res = cv2.bitwise_and(license_plate_crop, license_plate_crop, mask=mask_inv)
    res_white = cv2.bitwise_or(res, cv2.bitwise_and(white_background, white_background, mask=mask))
    return res_white


def adjust_license_plate_2(license_plate_crop):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2HSV)

    # Define range of black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])  # Adjust upper value for different shades of black

    # Threshold the HSV image to get only black colors
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Invert the mask to get everything except black
    mask_inv = cv2.bitwise_not(mask_black)

    # Create a white image of the same size as the frame
    white_background = np.full(license_plate_crop.shape, 255, dtype=np.uint8)

    # Use the inverted mask to replace the non-black areas with white
    res = cv2.bitwise_and(license_plate_crop, license_plate_crop, mask=mask_black)
    res_white = cv2.bitwise_or(res, cv2.bitwise_and(white_background, white_background, mask=mask_inv))

    return res_white


def get_license_plate_details(license_plate_crop_gray):
    _, license_plate_crop_lower_thresh = cv2.threshold(license_plate_crop_gray, 40, 255, cv2.THRESH_BINARY_INV)
    _, license_plate_crop_higher_thresh = cv2.threshold(license_plate_crop_gray, 138, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("low_thresh", license_plate_crop_lower_thresh)
    cv2.imshow("higher_thresh", license_plate_crop_higher_thresh)

    # read license plate number
    license_plate_text_lt, license_plate_text_score_lt = read_license_plate(license_plate_crop_lower_thresh)
    license_plate_text_ht, license_plate_text_score_ht = read_license_plate(license_plate_crop_higher_thresh)

    # returns license plate text based on higher score
    if license_plate_text_score_ht and license_plate_text_score_lt:
        (license_plate_text, license_plate_text_score) = (
            license_plate_text_ht, license_plate_text_score_ht) if str(license_plate_text_score_ht) > str(
            license_plate_text_score_lt) else (license_plate_text_lt, license_plate_text_score_lt)
    elif license_plate_text_score_ht:
        license_plate_text, license_plate_text_score = license_plate_text_ht, license_plate_text_score_ht
    else:
        license_plate_text, license_plate_text_score = license_plate_text_lt, license_plate_text_score_lt
    return license_plate_text, license_plate_text_score


def prepare_json_data(license_plate_text, license_plate_crop):
    lpr_uuid = str(uuid.uuid4())
    lpr_id = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    access_point = 0
    plate_text = license_plate_text

    success, encoded_image = cv2.imencode('.png', license_plate_crop)

    if success:
        base64_image = base64.b64encode(encoded_image).decode('utf-8')
        plate_image = base64_image

        data = {
            "lprUuid": lpr_uuid,
            "lprID": lpr_id,
            "accessPoint": access_point,
            "plateText": plate_text,
            "plateImage": plate_image
        }

        json_data = json.dumps(data, indent=4)
        return json_data
    else:
        print("Failed to encode image")


def post_request(data):
    url = 'https://192.168.1.1/api/lpr/licenseplate'
    response = requests.post(url, json=data)

    if response.status_code == 201:
        print('Data posted successfully')
        data = response.json()
        print(data)
    else:
        print(f"Failed to post data: {response.status_code}")
