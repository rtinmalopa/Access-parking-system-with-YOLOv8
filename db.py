import sqlite3
import cv2
import numpy as np
from utils import compare_images_ssim


def db_contains(license_plate_text, car_image):
    conn = sqlite3.connect('license_plates.db')
    c = conn.cursor()

    c.execute("SELECT license_plate, car_image FROM vehicles WHERE license_plate = ?", (license_plate_text,))
    result = c.fetchone()

    if result:
        if not result[1] is not None and car_image is not None:
            set_car_image_to_license_plate(conn, license_plate_text, car_image)
            conn.close()
            return True
        else:
            db_image = get_car_image_by_license_plate(c, result[0])
            if db_image is not None:
                score = compare_images_ssim(db_image, car_image)
                if score >= 0.20:
                    conn.close()
                    return True
            conn.close()
            return False

    conn.close()
    return False


def get_car_image_by_license_plate(c, license_plate_text):
    c.execute('''
    SELECT car_image
        FROM vehicles
            WHERE license_plate = ?
    ''', (license_plate_text,))

    result = c.fetchone()
    if result and result[0]:
        # Convert the binary data to a NumPy array
        image_data = np.frombuffer(result[0], dtype=np.uint8)

        # Decode the image from the NumPy array
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        return image
    return None


def set_car_image_to_license_plate(conn, license_plate, car_image):
    # Encode the image as a PNG file in memory
    _, buffer = cv2.imencode('.png', car_image)

    # Convert the buffer to binary data
    image_data = buffer.tobytes()

    conn.cursor().execute('''
    UPDATE vehicles
    SET car_image = ?
    WHERE license_plate = ?
    ''', (image_data, license_plate))

    conn.commit()


# def db_contains_lp(license_plate):
#     conn = sqlite3.connect('license_plates.db')
#     c = conn.cursor()
#
#     c.execute("SELECT license_plate FROM vehicles WHERE license_plate = ?", (license_plate,))
#     result = c.fetchone()
#
#     c.close()
#     return result is not None


# print(db_contains("MI399CT", None))
# print(get_car_image_by_license_plate("MI399CT"))

# conn = sqlite3.connect('license_plates.db')

# c = conn.cursor()


# c.execute("CREATE TABLE vehicles (license_plate TEXT NOT NULL UNIQUE, car_image BLOB)")
# c.execute("INSERT INTO vehicles VALUES ('MI399CT', NULL)")
# c.execute("INSERT INTO vehicles VALUES ('MI899FM', NULL)")
# c.execute("INSERT INTO vehicles VALUES ('MI589FN', NULL)")
# c.execute("INSERT INTO vehicles VALUES ('MI098EX', NULL)")

# c.execute("SELECT * FROM vehicles")
# print(c.fetchone())
# # conn.commit()
# print(db_contains('MI399CT'))

# conn.close()
