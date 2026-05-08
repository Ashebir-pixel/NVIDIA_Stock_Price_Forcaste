import mysql.connector
from mysql.connector import errorcode

db_config = {
    'user': 'root',
    'password': '2A6?7#4n6&3r', # Double check this!
    'host': '127.0.0.1',
    'port': 3306,
    'auth_plugin': 'mysql_native_password' # This forces the 'Legacy' mode we installed
}

try:
    print("Attempting Handshake...")
    conn = mysql.connector.connect(**db_config)
    print("SUCCESS: Connection established!")
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("ERROR: Something is wrong with your username or password.")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("ERROR: Database does not exist.")
    else:
        print(f"DETAILED ERROR: {err}")