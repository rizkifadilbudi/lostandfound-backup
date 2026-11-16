const express = require("express");
const multer = require("multer");
const FormData = require("form-data");
const axios = require("axios"); // ← tambah ini
require("dotenv").config();

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

app.post("/api/classify", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "Harap unggah gambar" });
    }

    const formData = new FormData();
    formData.append("file", req.file.buffer, {
      filename: req.file.originalname || "image.jpg",
      contentType: req.file.mimetype || "image/jpeg",
    });

    // ← Ganti fetch() dengan axios — kompatibel dengan form-data npm package
    const aiResponse = await axios.post(
      process.env.AI_SERVICE_URL || "http://localhost:8000/predict",
      formData,
      { headers: formData.getHeaders() },
    );

    return res.json(aiResponse.data);
  } catch (error) {
    if (error.response) {
      // FastAPI mengembalikan error
      console.error("FastAPI Error Response:", error.response.data);
      return res
        .status(error.response.status)
        .json({ error: "Gagal diproses oleh AI Service" });
    }
    console.error("Error connecting to AI Service:", error.message);
    return res.status(500).json({ error: "Gagal terhubung ke layanan AI" });
  }
});

app.listen(3000, () => console.log("Backend running on port 3000"));
