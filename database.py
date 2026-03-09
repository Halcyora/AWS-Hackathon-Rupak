import sqlite3
import time

def init_db():
    """Initializes the local SQLite DB for edge computing."""
    conn = sqlite3.connect('fractallens_edge.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnostic_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            fractal_dimension REAL,
            prediction_class TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            synced BOOLEAN DEFAULT FALSE
        )
    ''')
    conn.commit()
    conn.close()

def save_result(patient_name, fd_value, prediction):
    """Saves a diagnostic result locally."""
    conn = sqlite3.connect('fractallens_edge.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO diagnostic_results (patient_name, fractal_dimension, prediction_class)
        VALUES (?, ?, ?)
    ''', (patient_name, fd_value, prediction))
    conn.commit()
    conn.close()
    
def get_unsynced_count():
    conn = sqlite3.connect('fractallens_edge.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM diagnostic_results WHERE synced = FALSE')
    count = cursor.fetchone()[0]
    conn.close()
    return count