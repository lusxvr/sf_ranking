import sqlite3
import json

def extract_votes(db_path, json_path):
    # Connect to the SQLite database
    connection = sqlite3.connect("votes.db")
    cursor = connection.cursor()

    # Query the numbers you need
    cursor.execute("SELECT * FROM votes")
    numbers = cursor.fetchall()  # Fetch all rows

    # Convert to a list of numbers
    votes = {}
    for row in numbers:
        votes[row[0]] = row[1]

    # Save to JSON
    with open("votes.json", "w") as json_file:
        json.dump(votes, json_file)

    # Close the connection
    connection.close()
