import psycopg2
from psycopg2 import sql

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    user="postgres",
    password="postgres",
    database="postgres"
)

conn.autocommit = True
cursor = conn.cursor()

# Create database
try:
    cursor.execute("CREATE DATABASE analytics_db")
    print("✅ Database 'analytics_db' created!")
except:
    print("⚠️ Database already exists")

cursor.close()
conn.close()

# Connect to new database
conn = psycopg2.connect(
    host="localhost",
    port="5432",
    user="postgres",
    password="postgres",
    database="analytics_db"
)

cursor = conn.cursor()

# Create sales table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        id SERIAL PRIMARY KEY,
        date DATE,
        product_name TEXT,
        quantity INTEGER,
        price DECIMAL(10,2),
        revenue DECIMAL(10,2),
        region TEXT
    )
""")

# Clear old data
cursor.execute("DELETE FROM sales")

# Insert sample data
cursor.execute("""
    INSERT INTO sales (date, product_name, quantity, price, revenue, region) VALUES
    ('2024-01-15', 'Laptop', 5, 800.00, 4000.00, 'North'),
    ('2024-01-20', 'Mouse', 100, 25.00, 2500.00, 'South'),
    ('2024-02-10', 'Keyboard', 50, 45.00, 2250.00, 'East'),
    ('2024-02-15', 'Monitor', 20, 300.00, 6000.00, 'West'),
    ('2024-03-05', 'Laptop', 3, 800.00, 2400.00, 'North')
""")

conn.commit()
print("✅ Sales table created with data!")

cursor.close()
conn.close()
print("🎉 PostgreSQL is ready!")