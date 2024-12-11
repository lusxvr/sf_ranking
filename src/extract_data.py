import sqlite3
import json

<<<<<<< HEAD
# Connectto SQLite database
connection = sqlite3.connect("votes.db")
cursor = connection.cursor()

#Query
cursor.execute("SELECT * FROM votes")
numbers = cursor.fetchall()  # Fetch all rows

votes = {}
for row in numbers:
    votes[row[0]] = row[1]
#numbers_list = [row[0] for row in numbers]

#save to JSON
with open("votes.json", "w") as json_file:
    json.dump(votes, json_file)

connection.close()
=======
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
    #numbers_list = [row[0] for row in numbers]

    # Save to JSON
    with open("votes.json", "w") as json_file:
        json.dump(votes, json_file)

    # Close the connection
    connection.close()
>>>>>>> 8c9b34f8f97a0bb6aaf16534a9cfa4c4ed715f72
