import mysql.connector

class MySQLDB:
    def __init__(self, host, user, password, database):
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_name VARCHAR(255),
            prediction VARCHAR(50),
            model_output INT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.cursor.execute(query)
        self.conn.commit()

    def insert_prediction(self, image_name, prediction, model_output):
        query = """
        INSERT INTO predictions (image_name, prediction, model_output)
        VALUES (%s, %s, %s)
        """
        self.cursor.execute(
            query,
            (image_name, prediction, model_output)
        )
        self.conn.commit()
