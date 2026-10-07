const express = require('express');
const axios = require('axios');
const bodyParser = require('body-parser');
const path = require('path');

const app = express();
const PORT = 3000;
const FLASK_SERVER_URL = 'http://127.0.0.1:5050'; 

app.use(express.static(path.join(__dirname, 'public')));
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));

app.post('/fetch', async (req, res) => {
    const { query, presentdate, max_records } = req.body;
    console.log(presentdate);
    try {
        const response = await axios.post(`${FLASK_SERVER_URL}/fetch_articles`, {
            query, presentdate, max_records: Number(max_records)
        });
        res.json(response.data);
    } catch (error) {
        console.error('Error in /fetch route:', error);
        res.status(500).send(error.toString());
    }
});

app.post('/cluster', async (req, res) => {
    const { new_papers, old_papers, sim_threshold} = req.body;
    try {
        const response = await axios.post(`${FLASK_SERVER_URL}/cluster_papers`, {
            new_papers, old_papers, sim_threshold
        });
        res.json(response.data);
    } catch (error) {
        console.error('Error in /cluster route:', error);
        res.status(500).send(error.toString());
    }
});

app.post('/generate_cluster_names', async (req, res) => {
    const { clusters } = req.body;
    try {
        const response = await axios.post(`${FLASK_SERVER_URL}/generate_cluster_names`, {
            clusters
        });
        res.json(response.data);
    } catch (error) {
        console.error('Error in /generate_cluster_names route:', error);
        res.status(500).send(error.toString());
    }
});

app.post('/identify_trends', async (req, res) => {
    const { clusters } = req.body;
    try {
        const response = await axios.post(`${FLASK_SERVER_URL}/identify_trends`, {
            clusters
        });
        res.json(response.data);
    } catch (error) {
        console.error('Error in /identify_trends route:', error);
        res.status(500).send(error.toString());
    }
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});