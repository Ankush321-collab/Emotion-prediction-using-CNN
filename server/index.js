import express from "express";
import fetch from "node-fetch";
import cors from "cors";

const app = express();

// Allow frontend requests
app.use(cors());
app.use(express.json());

// Endpoint that React will call
app.post("/analyze", async (req, res) => {
  try {
    const flaskResponse = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body)
    });

    const data = await flaskResponse.json();
    res.json(data);
  } catch (error) {
    console.error("Error contacting Flask API:", error);
    res.status(500).json({ error: "Error contacting Flask API" });
  }
});

// Default route
app.get("/", (req, res) => {
  res.send("Node middleware is running!");
});

const PORT = 4000;
app.listen(PORT, () => console.log(`✅ Node server running on port ${PORT}`));

