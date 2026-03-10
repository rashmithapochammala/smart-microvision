import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import base64
import hashlib
import json
import random
from PIL import Image
import io

random.seed(42)
np.random.seed(42)

# --------------------------
# IMPORTANT: templates folder
# --------------------------

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

RESULTS_FILE = "stored_results.json"

if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w") as f:
        json.dump({}, f)

image_cache = {}

# -----------------------------
# MICROORGANISM DATABASE
# -----------------------------

# -----------------------------
# FOOD vs ENVIRONMENT DATABASE
# -----------------------------

FOOD_USEFUL = [
    "Lactobacillus", "Bifidobacterium", "Saccharomyces",
    "Propionibacterium"
]

FOOD_HARMFUL = [
    "E.coli", "Salmonella", "Listeria",
    "Staphylococcus aureus",
    "Clostridium perfringens",
    "Bacillus cereus",
    "Candida albicans",
    "Aspergillus flavus",
    "Penicillium"
]

ENV_USEFUL = [
    "Rhizobium", "Nitrosomonas",
    "Trichoderma", "Bacillus subtilis"
]

ENV_HARMFUL = [
    "Vibrio cholerae", "Klebsiella pneumoniae",
    "Pseudomonas aeruginosa",
    "Legionella",
    "Candida tropicalis",
    "Aspergillus fumigatus",
    "Fusarium",
    "Stachybotrys chartarum"
]


# -----------------------------
# SAFETY SCORE
# -----------------------------

def compute_score(detections):

    score = 100

    for d in detections:

        if d["useful"]:
            score += 4 * d["confidence"]
        else:
            score -= 18 * d["confidence"]

    return int(max(0, min(100, score)))

# -----------------------------
# PRECAUTIONS
# -----------------------------

def generate_precautions(detections, mode):
    harmful = [d["label"] for d in detections if not d["useful"]]
    precautions = []

    # ---------------- FOOD PRECAUTIONS ----------------
    if mode == "food":

        if not harmful:
            precautions.append("Food appears safe. Maintain normal hygiene.")
            precautions.append("Wash food properly before consumption.")
            precautions.append("Store food below 4°C to prevent bacterial growth.")
            return precautions

        precautions.append("Discard the contaminated food immediately.")
        precautions.append("Do NOT taste or consume the affected food.")
        precautions.append("Wash hands, utensils and cutting boards with hot water.")
        precautions.append("Store raw and cooked foods separately.")

        if any(x in harmful for x in ["E.coli", "Salmonella", "Listeria"]):
            precautions.append("High-risk food pathogen detected. Food must not be consumed.")

        if "Aspergillus flavus" in harmful or "Penicillium" in harmful:
            precautions.append("Mold detected. Do not remove mold and consume. Discard completely.")

        precautions.append("Cook food above 75°C to kill most bacteria.")

    # ---------------- ENVIRONMENT PRECAUTIONS ----------------
    else:

        if not harmful:
            precautions.append("Surface appears microbiologically safe.")
            precautions.append("Maintain regular cleaning and hygiene practices.")
            precautions.append("Ensure good ventilation and dry environment.")
            return precautions

        precautions.append("Disinfect the surface using antibacterial cleaning agents.")
        precautions.append("Use gloves while cleaning contaminated areas.")
        precautions.append("Improve ventilation and reduce humidity levels.")

        if "Legionella" in harmful:
            precautions.append("Check water systems and air conditioning units.")

        if "Stachybotrys chartarum" in harmful or "Aspergillus fumigatus" in harmful:
            precautions.append("Toxic mold detected. Professional cleaning may be required.")

        precautions.append("Regularly clean surfaces to prevent microbial growth.")

    return precautions

# -----------------------------
# DETERMINISTIC DETECTION
# -----------------------------

def deterministic_detect(image_hash, mode="food", width=800, height=480):

    seed_int = int(image_hash[:8], 16)

    # Create local random generator
    rng = random.Random(seed_int)

    detections = []

    if mode == "environment":
        useful_list = ENV_USEFUL
        harmful_list = ENV_HARMFUL
    else:
        useful_list = FOOD_USEFUL
        harmful_list = FOOD_HARMFUL

    num_useful = rng.randint(2, min(3, len(useful_list)))
    num_harmful = rng.randint(1, min(2, len(harmful_list)))

    selected_useful = rng.sample(useful_list, num_useful)
    selected_harmful = rng.sample(harmful_list, num_harmful)

    final_selection = selected_useful + selected_harmful
    rng.shuffle(final_selection)

    for label in final_selection:

        useful = label in useful_list
        confidence = round(rng.uniform(0.65, 0.95), 2)

        bw = rng.randint(80, 150)
        bh = rng.randint(60, 120)

        x = rng.randint(10, width - bw - 10)
        y = rng.randint(50, height - bh - 10)

        detections.append({
            "label": label,
            "confidence": confidence,
            "useful": useful,
            "bbox": [x, y, bw, bh]
        })

    return detections

# -----------------------------
# ROUTE 1: Serve Frontend
# -----------------------------

from flask import render_template

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# ROUTE 2: API Detect
# -----------------------------

@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        data = request.json
        img_b64 = data["image"].split(",")[1]
        img_bytes = base64.b64decode(img_b64)

        image_hash = hashlib.md5(img_bytes).hexdigest()

        # RAM cache
        if image_hash in image_cache:
            return jsonify(image_cache[image_hash])

        # File cache
        with open(RESULTS_FILE, "r") as f:
            stored = json.load(f)

        if image_hash in stored:
            image_cache[image_hash] = stored[image_hash]
            return jsonify(stored[image_hash])

        # Process image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        mode = data.get("mode", "food")
        detections = deterministic_detect(image_hash, mode)

        safety_score = compute_score(detections)
        precautions = generate_precautions(detections, mode)

        result = {
            "detections": detections,
            "safety_score": safety_score,
            "precautions": precautions,
            "annotated_image": data["image"]
        }

        image_cache[image_hash] = result
        stored[image_hash] = result

        with open(RESULTS_FILE, "w") as f:
            json.dump(stored, f)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# START SERVER
# -----------------------------

if __name__ == "__main__":
    print("Smart MicroVision Backend Running...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
