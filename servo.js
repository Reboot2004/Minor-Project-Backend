const express = require('express');
const serveIndex = require('serve-index');
const cors = require('cors');
const path = require('path');

const app = express();
const port = 8000;
const directoryToServe = path.join(__dirname,"/");
const corsOptions = {
    origin: ['http://localhost:3000'],
    methods: 'GET, POST, PUT, DELETE',
    credentials: true,
    allowedHeaders: 'Content-Type, Authorization',
};
// Enable CORS for all routes
app.use(cors());
console.log(directoryToServe)

// Serve static files and directory listing
app.use('/', express.static(directoryToServe), serveIndex(directoryToServe, { icons: true }));
app.listen(port, () => {
    console.log(`File server running at http://localhost:${port}/`);
});