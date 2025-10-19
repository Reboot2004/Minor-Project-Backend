from flask import Flask, jsonify, request, send_file, render_template
from flask_cors import CORS
from lrp_pipeline_2 import lrp_main
from utils import create_folders, delete_folders, create_zip_file
from cam_pipeline import cam_main, cam_process_single_image
import os
import base64

app = Flask(__name__)
CORS(app)


@app.route("/api/upload", methods=["GET"])
def get_data():
    data = {"message": "Hello from Flask backend!"}
    return jsonify(data)


@app.route("/api/upload", methods=["POST"])
def submit_data():
    # first clear all the existing files in uploads, heatmaps, segmentations, tables, cell_descriptors folders
    folder_names = [
        "uploads",
        "heatmaps",
        "segmentations",
        "tables",
        "cell_descriptors",
    ]
    delete_folders(folder_names)
    create_folders(folder_names)

    # Ensure the uploads directory exists
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # then upload the submitted file(s)
    file = list(dict(request.files).values())[0]
    print(file)
    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)  # Save to 'uploads' directory

    # Process data here
    return jsonify({
        "message": "Data received successfully!",
        "file_path": file_path
    })


@app.route("/api/inputform", methods=["POST"])
def submit_form():
    data = dict(request.json)  # format of data: {'model': 'VGGNet', 'xaiMethod': 'LRP'}
    print(data)
    
    # Check if we have images in the uploads directory
    uploads_dir = "uploads"
    image_files = [f for f in os.listdir(uploads_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) 
                   and not f.startswith('.')]
    
    if not image_files:
        return jsonify({"error": "No images found in uploads directory"}), 400
        
    # Process the first image (or all images based on your requirements)
    image_path = os.path.join(uploads_dir, image_files[0])
    
    if "LRP" in data["xaiMethod"]:
        result_dict = lrp_main(float(data["magval"]))
        # Extract relevant results to show in the frontend
        return jsonify({
            "success": True,
            "summary": f"LRP analysis completed with magnification {data['magval']}",
            "details": "Nucleus and cytoplasm segmented successfully",
            "results": result_dict
        })
        
    elif "GradCAM++" in data["xaiMethod"]:
        # Process single image with GradCAM++
        result_dict, output_paths = cam_process_single_image(image_path, float(data["magval"]))
        
        # Read and encode the output files for display
        original_image = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
        heatmap_image = base64.b64encode(open(output_paths["heatmap"], "rb").read()).decode("utf-8")
        mask_image = base64.b64encode(open(output_paths["mask"], "rb").read()).decode("utf-8")
        table_image = base64.b64encode(open(output_paths["table"], "rb").read()).decode("utf-8")
        
        return jsonify({
            "success": True,
            "summary": f"GradCAM++ analysis completed with magnification {data['magval']}",
            "details": "Nucleus and cytoplasm segmented successfully",
            "results": {
                "originalImage": original_image,
                "heatmapImage": heatmap_image,
                "maskImage": mask_image,
                "tableImage": table_image
            }
        })


@app.route("/api/zip", methods=["GET"])
def get_csv():
    create_zip_file()
    return send_file("outputs.zip", as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
