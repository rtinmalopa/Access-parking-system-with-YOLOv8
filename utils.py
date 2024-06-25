import string
import easyocr
import cv2
import numpy as np
import base64
import json
import uuid
import requests
from datetime import datetime
from skimage.metrics import structural_similarity as ssim

# Initialize the OCR reader
reader = easyocr.Reader(['en', 'sk'])

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


def get_car(license_plate, vehicles):
    x1, y1, x2, y2, score, class_id = license_plate

    for i in range(len(vehicles)):
        car_x1, car_y1, car_x2, car_y2, track_id, _ = vehicles[i]

        if x1 > car_x1 and y1 > car_y1 and x2 < car_x2 and y2 < car_y2:
            return vehicles[i]

    return -1, -1, -1, -1, -1, -1


def process_license_plate(license_plate_crop):
    resized_image = resize_image(license_plate_crop, 300, 100)
    # cv2.imshow("resized_image", resized_image)

    adjusted = adjust_license_plate(resized_image)
    # cv2.imshow("adjusted", adjusted)

    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)

    alpha = 2.0
    beta = 0
    contrasted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    cv2.imshow("contrasted", contrasted)

    license_plate_text, license_plate_text_score = read_license_plate(contrasted)
    return license_plate_text, license_plate_text_score


def resize_image(image, target_width, target_height):
    h, w = image.shape[:2]
    if w > h:
        new_w = target_width
        new_h = int((h / w) * target_width)
    else:
        new_h = target_height
        new_w = int((w / h) * target_height)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized_image


def adjust_license_plate(license_plate_crop):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_dark_blue = np.array([100, 150, 0])
    upper_dark_blue = np.array([140, 255, 100])
    lower_light_blue = np.array([90, 50, 50])
    upper_light_blue = np.array([110, 255, 255])

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
    mask_light_blue = cv2.inRange(hsv, lower_light_blue, upper_light_blue)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_dark_red = cv2.inRange(hsv, lower_dark_red, upper_dark_red)

    mask_blue = cv2.bitwise_or(mask_blue1, mask_blue2)
    mask_blue = cv2.bitwise_or(mask_blue, mask_light_blue)
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


def read_license_plate(license_plate_crop):
    custom_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    detections = reader.readtext(license_plate_crop, allowlist=custom_characters, width_ths=0.9, min_size=10, text_threshold=0.7, low_text=0.5, link_threshold=0.7)

    # cv2.imshow('lp', license_plate_crop)
    region_threshold = 0.3
    license_plate_text, license_plate_score = filter_text(license_plate_crop, detections, region_threshold)

    if license_plate_text and license_plate_score:
        return license_plate_text, license_plate_score
        # license_complies_format(license_plate_text):
        #     return format_license(license_plate_text), license_plate_score

    return None, None


def filter_text(license_plate_crop, ocr_result, region_threshold):
    rectangle_size = license_plate_crop.shape[0] * license_plate_crop.shape[1]

    for result in ocr_result:
        width = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        # print(length, height)
        if width * height / rectangle_size > region_threshold:
            plate = result[1].replace(" ", "")
            score = result[2]
            return plate, score
    return None, None


def display_license_plate_text(frame, license_plate, license_plate_text):
    x1, y1, x2, y2, _, _ = license_plate

    # Calculate the width of the license plate
    plate_width = x2 - x1

    # Define the initial font and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 4

    # Calculate the text size with the initial font scale
    text_size, _ = cv2.getTextSize(license_plate_text, font, font_scale, font_thickness)

    # Adjust the font scale so that the text width matches the plate width
    font_scale = plate_width / text_size[0]

    # Recalculate the text size with the adjusted font scale
    text_size, _ = cv2.getTextSize(license_plate_text, font, font_scale, font_thickness)

    # Calculate text position (above the cropped image)
    text_x = int(x1)
    text_y = int(y1) - 10  # 10 pixels above the license plate

    # Define padding around the text
    padding = 10

    # Calculate the rectangle coordinates for the text background with padding
    rect_top_left = (text_x - padding, text_y - text_size[1] - padding)
    rect_bottom_right = (text_x + text_size[0] + padding, text_y + padding)

    # Check if text and background fits within the frame
    if rect_bottom_right[0] <= frame.shape[1] and rect_top_left[1] >= 0:
        # Draw the white background rectangle with padding
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, (255, 255, 255), thickness=cv2.FILLED)

        # Draw the text on the white background
        cv2.putText(frame, license_plate_text, (text_x, text_y), font, font_scale, (0, 255, 0), font_thickness)


def resize_images_to_same_dimensions(image1, image2):
    img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Resize img2 to match img1 dimensions
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return img1, img2_resized


# Function to compute SSIM between two images
def compare_images_ssim(image1, image2):
    img1, img2 = resize_images_to_same_dimensions(image1, image2)
    score, diff = ssim(img1, img2, full=True)
    return score


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
