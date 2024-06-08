import sqlite3


def db_contains(license_plate_text):
    conn = sqlite3.connect('license_plates.db')
    c = conn.cursor()

    c.execute("SELECT * FROM license_plates WHERE number = ?", (license_plate_text,))
    result = c.fetchone()

    conn.close()

    return result is not None

# conn = sqlite3.connect('license_plates.db')
#
# c = conn.cursor()


# c.execute("CREATE TABLE license_plates ( number text )")
# c.execute("INSERT INTO license_plates VALUES ('MI589FN')")

# c.execute("SELECT * FROM license_plates")

# conn.commit()
# print(db_contains('MI399CT'))
#
# conn.close()
