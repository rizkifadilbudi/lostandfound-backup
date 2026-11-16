from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLOWorld
from PIL import Image, ImageEnhance
import io

app = FastAPI(title="Lost & Found AI Service")

# Muat YOLO-World — open-vocabulary detector (gratis, open-source)
print("Loading YOLO-World model...")
model = YOLOWorld("yolov8s-worldv2.pt")  # download otomatis dari Ultralytics

# Daftar kelas barang yang relevan untuk Lost & Found
# Tambah / hapus sesuai kebutuhan
LOST_FOUND_CLASSES = [
    "wallet", "phone", "smartphone", "laptop", "tablet",
    "bag", "backpack", "handbag", "suitcase", "umbrella",
    "glasses", "sunglasses", "keys", "watch", "helmet",
    "notebook", "book", "charger", "headphones", "earphones",
    "bottle", "water bottle", "jacket", "coat", "shoes",
    "hat", "cap", "scarf", "gloves", "belt",
    "camera", "mouse", "keyboard", "hard disk", "usb",
    "card", "id card", "passport", "necklace", "ring",
    "bracelet", "earring", "toy", "stuffed animal", "pen"
]

model.set_classes(LOST_FOUND_CLASSES)
print(f"YOLO-World ready! Mendeteksi {len(LOST_FOUND_CLASSES)} kelas barang hilang.")


def preprocess_image(image: Image.Image) -> Image.Image:
    """Tingkatkan kualitas gambar sebelum deteksi."""
    # Pastikan mode RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Tingkatkan kontras
    image = ImageEnhance.Contrast(image).enhance(1.3)
    # Tingkatkan ketajaman
    image = ImageEnhance.Sharpness(image).enhance(1.5)
    return image


@app.get("/")
def read_root():
    return {
        "status": "AI Service Running",
        "model": "YOLO-World (yolov8s-worldv2)",
        "total_classes": len(LOST_FOUND_CLASSES),
        "classes": LOST_FOUND_CLASSES
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validasi tipe file
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        filename = file.filename or ""
        if not any(filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]):
            raise HTTPException(status_code=400, detail="File harus berupa gambar (jpg, jpeg, png, gif, bmp, webp)")

    # Baca file gambar
    image_bytes = await file.read()

    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File tidak dapat dibaca sebagai gambar: {str(e)}")

    # Preprocessing untuk meningkatkan kualitas deteksi
    image = preprocess_image(image)

    # Jalankan deteksi objek dengan resolusi lebih tinggi
    try:
        results = model(image, imgsz=1280)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference gagal: {str(e)}")

    predictions = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Ambil prediksi dengan confidence > 25% (lebih sensitif dari sebelumnya)
            if confidence > 0.25:
                predictions.append({
                    "label": LOST_FOUND_CLASSES[class_id] if class_id < len(LOST_FOUND_CLASSES) else str(class_id),
                    "confidence": round(confidence, 2),
                    "bbox": {
                        "x1": round(float(box.xyxy[0][0]), 1),
                        "y1": round(float(box.xyxy[0][1]), 1),
                        "x2": round(float(box.xyxy[0][2]), 1),
                        "y2": round(float(box.xyxy[0][3]), 1)
                    }
                })

    # Urutkan dari confidence tertinggi
    predictions.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "success": True,
        "count": len(predictions),
        "predictions": predictions
    }