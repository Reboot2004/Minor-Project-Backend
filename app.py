from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from lrp_pipeline_2 import lrp_main
from cam_pipeline import cam_process_single_image
from utils import create_folders, delete_folders, create_zip_file
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import os
import base64

app = Flask(__name__)
CORS(app)

# === MongoDB Atlas Setup (Hugging Face Secret) ===
MONGO_URI = os.getenv("MONGO_URI")  # Add this secret in Hugging Face: Settings → Variables and secrets

if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set. Please add it in Hugging Face Space Secrets.")

client = MongoClient(MONGO_URI)
db = client["xai_results"]

try:
    client.admin.command("ping")
    print("✅ Connected to MongoDB Atlas successfully.")
except Exception as e:
    print("⚠️ MongoDB connection failed:", e)


# === ROUTE: Upload image ===
@app.route("/api/upload", methods=["POST"])
def submit_data():
    folder_names = ["uploads", "heatmaps", "segmentations", "tables", "cell_descriptors"]
    delete_folders(folder_names)
    create_folders(folder_names)

    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    file = list(dict(request.files).values())[0]
    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)

    return jsonify({
        "message": "Data received successfully!",
        "file_path": file_path
    })


# === ROUTE: Process input form (LRP or GradCAM++) ===
@app.route("/api/inputform", methods=["POST"])
def submit_form():
    data = dict(request.json)
    uploads_dir = "uploads"

    image_files = [f for f in os.listdir(uploads_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and not f.startswith('.')]

    if not image_files:
        return jsonify({"error": "No images found in uploads directory"}), 400

    image_path = os.path.join(uploads_dir, image_files[0])
    xai_method = data.get("xaiMethod", "Unknown")
    magval = float(data.get("magval", 1.0))

    # === LRP ===
    if "LRP" in xai_method:
        result_dict = lrp_main(magval)
        record = {
            "model": data.get("model"),
            "xaiMethod": xai_method,
            "magnification": magval,
            "classification": result_dict["classification"],
            "images": {
                "originalImage": result_dict["image1"],
                "heatmapImage": result_dict["inter1"],
                "maskImage": result_dict["mask1"],
                "tableImage": result_dict["table1"]
            },
            "timestamp": datetime.utcnow()
        }
        db.predictions.insert_one(record)
        return jsonify({
            "success": True,
            "summary": f"LRP completed with magnification {magval}",
            "classification": record["classification"],
            "results": record["images"]
        })

    # === GradCAM++ ===
    elif "GradCAM++" in xai_method:
        result_dict, output_paths = cam_process_single_image(image_path, magval)

        def encode_img(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        original = encode_img(image_path)
        heatmap = encode_img(output_paths["heatmap"])
        mask = encode_img(output_paths["mask"])
        table = encode_img(output_paths["table"])

        record = {
            "model": data.get("model"),
            "xaiMethod": xai_method,
            "magnification": magval,
            "classification": result_dict.get("class1"),
            "images": {
                "originalImage": original,
                "heatmapImage": heatmap,
                "maskImage": mask,
                "tableImage": table
            },
            "timestamp": datetime.utcnow()
        }

        db.predictions.insert_one(record)
        return jsonify({
            "success": True,
            "summary": f"GradCAM++ completed with magnification {magval}",
            "classification": record["classification"],
            "results": record["images"]
        })

    else:
        return jsonify({"error": "Invalid XAI method"}), 400


# === ROUTE: Create ZIP (optional) ===
@app.route("/api/zip", methods=["GET"])
def get_csv():
    zip_path = "outputs.zip"
    create_zip_file()

    if not os.path.exists(zip_path):
        return jsonify({"error": "outputs.zip not found"}), 404

    return send_file(zip_path, as_attachment=True)


# === ROUTE: Fetch all previous predictions ===
@app.route("/api/oldpreds", methods=["GET"])
def list_old_predictions():
    preds = list(db.predictions.find().sort("timestamp", -1))
    result = []
    for p in preds:
        result.append({
            "id": str(p["_id"]),
            "model": p.get("model"),
            "xaiMethod": p.get("xaiMethod"),
            "magnification": p.get("magnification"),
            "classification": p.get("classification"),
            "images": p.get("images"),
            "timestamp": p["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        })
    return jsonify(result)


# === ROUTE: Fetch one old prediction by ID ===
@app.route("/api/oldpreds/<id>", methods=["GET"])
def get_old_prediction(id):
    try:
        record = db.predictions.find_one({"_id": ObjectId(id)})
        if not record:
            return jsonify({"error": "Record not found"}), 404
        record["_id"] = str(record["_id"])
        record["timestamp"] = record["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        return jsonify(record)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask XAI API running successfully"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)