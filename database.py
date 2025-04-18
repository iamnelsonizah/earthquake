# database.py
import sqlite3
import pandas as pd

def init_db(csv_path):
    try:
        # Connect to SQLite database
        conn = sqlite3.connect('earthquakes.db')
        # Load specified columns from CSV
        df = pd.read_csv(csv_path, usecols=['Longitude', 'Latitude', 'Depth', 'Country', 'Magnitude'])
        # Ensure numeric data types
        df[['Longitude', 'Latitude', 'Depth', 'Magnitude']] = df[['Longitude', 'Latitude', 'Depth', 'Magnitude']].apply(pd.to_numeric, errors='coerce')
        # Drop rows with missing values
        df = df.dropna()
        # Store in SQLite
        df.to_sql('earthquakes', conn, if_exists='replace', index=False)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def get_data(query="SELECT * FROM earthquakes"):
    try:
        conn = sqlite3.connect('earthquakes.db')
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error querying database: {e}")
        return pd.DataFrame()