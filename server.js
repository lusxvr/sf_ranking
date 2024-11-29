// server.js
const express = require('express');
const path = require('path');
const bodyParser = require('body-parser');
const fs = require('fs');
const sqlite3 = require('sqlite3').verbose();

const app = express();
const PORT = process.env.PORT || 3000;

// Initialize SQLite database
const db = new sqlite3.Database(':memory:');

db.serialize(() => {
    db.run("CREATE TABLE IF NOT EXISTS votes (image TEXT PRIMARY KEY, count INTEGER DEFAULT 0)", (err) => {
        if (err) {
            console.error('Error creating votes table:', err);
        } else {
            console.log('Votes table initialized');
        }
    });
});

console.log('Starting server...');

// Middleware to handle JSON requests
app.use(bodyParser.json());

// Serve static files from the root directory
app.use(express.static(path.join(__dirname)));

// Add logging middleware for all requests
app.use((req, res, next) => {
    console.log(`${req.method} ${req.url}`);
    next();
});

// Serve data files with detailed logging
app.use('/data', (req, res, next) => {
    console.log('Data request:', req.url);
    const filePath = path.join(__dirname, 'data', req.url);
    if (fs.existsSync(filePath)) {
        res.sendFile(filePath);
    } else {
        res.status(404).send('File not found');
    }
});

// Endpoint to get images
app.get('/api/images', (req, res) => {
    const images = JSON.parse(fs.readFileSync(path.join(__dirname, 'data', 'images.json')));
    res.json(images);
});

// Endpoint to vote for an image
app.post('/api/vote', (req, res) => {
    const { image } = req.body;
    if (!image) {
        console.error('Vote error: Image is required');
        return res.status(400).send('Image is required');
    }
    
    console.log('Voting for image:', image); // Log the image being voted for

    db.run("INSERT INTO votes (image, count) VALUES (?, 1) ON CONFLICT(image) DO UPDATE SET count = count + 1", [image], function(err) {
        if (err) {
            console.error('Database error during vote:', err); // Log the error
            return res.status(500).send('Database error');
        }
        console.log('Vote recorded for image:', image); // Log successful vote
        res.status(200).send('Vote received');
    });
});

// Endpoint to get vote counts
app.get('/api/votes', (req, res) => {
    db.all("SELECT image, count FROM votes", (err, rows) => {
        if (err) {
            return res.status(500).send('Database error');
        }
        res.json(rows);
    });
});

// Serve index.html for the root route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
