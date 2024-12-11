import sqlite3
import json

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
