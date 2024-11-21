// server.js
const express = require('express');
const bodyParser = require('body-parser');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3000;

console.log('Starting server...');

// Log the current directory and data directory path
const dataPath = path.join(__dirname, '../data');
console.log('Current directory:', __dirname);
console.log('Data directory path:', dataPath);

// Check if data directory exists
if (!fs.existsSync(dataPath)) {
    console.error('Error: Data directory not found at:', dataPath);
    process.exit(1);
}

// List files in data directory
fs.readdir(dataPath, (err, files) => {
    if (err) {
        console.error('Error reading data directory:', err);
    } else {
        console.log('Files in data directory:', files);
    }
});

// Global error handling middleware
app.use((err, req, res, next) => {
    console.error('Global error:', err);
    res.status(500).json({ error: err.message });
});

app.use(bodyParser.json());

// Serve static files from public directory
app.use(express.static(path.join(__dirname, 'public')));

// Add logging middleware for all requests
app.use((req, res, next) => {
    console.log(`${req.method} ${req.url}`);
    next();
});

// Serve data files with detailed logging
app.use('/data', (req, res, next) => {
    console.log('Data request:', req.url);
    const filePath = path.join(dataPath, req.url);
    console.log('Looking for file:', filePath);
    if (fs.existsSync(filePath)) {
        console.log('File exists, serving:', filePath);
        res.sendFile(filePath);
    } else {
        console.log('File not found:', filePath);
        next();
    }
});

// Add CSP headers middleware
app.use((req, res, next) => {
    res.setHeader(
        'Content-Security-Policy',
        "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net"
    );
    next();
});

const db = new sqlite3.Database(':memory:');
db.serialize(() => {
    db.run(`CREATE TABLE votes (
        image TEXT PRIMARY KEY,
        shown_count INTEGER DEFAULT 0,
        liked_count INTEGER DEFAULT 0
    )`);
});

// API to record a vote
app.post('/vote', (req, res) => {
    try {
        const { selectedImage } = req.body;
        if (!selectedImage) {
            console.error('No image specified in request');
            return res.status(400).json({ error: 'Image not specified' });
        }

        // Log the full request body and image path
        console.log('Full request body:', req.body);
        console.log('Recording vote for:', selectedImage);

        // Extract just the filename from the path
        const imageName = selectedImage.split('/').pop();
        console.log('Image name:', imageName);

        db.run(
            `INSERT INTO votes (image, shown_count, liked_count)
             VALUES (?, 1, 1)
             ON CONFLICT(image)
             DO UPDATE SET shown_count = shown_count + 1, liked_count = liked_count + 1`,
            [imageName],
            function(err) {
                if (err) {
                    console.error('Database error details:', err);
                    return res.status(500).json({ error: err.message });
                }
                console.log('Vote successfully recorded. Changes:', this.changes);
                res.json({ 
                    success: true,
                    message: 'Vote recorded', 
                    changes: this.changes 
                });
            }
        );
    } catch (error) {
        console.error('Server error:', error);
        res.status(500).json({ error: error.message });
    }
});

app.get('/stats', (req, res) => {
    db.all("SELECT * FROM votes", [], (err, rows) => {
        if (err) return res.status(500).send('Database error');
        res.json(rows);
    });
});

// Add a test endpoint to check database state
app.get('/debug/votes', (req, res) => {
    db.all("SELECT * FROM votes", [], (err, rows) => {
        if (err) {
            console.error('Debug endpoint error:', err);
            return res.status(500).send(`Database error: ${err.message}`);
        }
        res.json(rows);
    });
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});