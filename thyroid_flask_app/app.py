from flask import Flask, render_template, request
import joblib
import pdfplumber
import os
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"

# Load trained model (use joblib to load .joblib safely)
MODEL_PATH = "thyroid_model_weighted_19features.joblib"  # if your file has different name, change here
model = joblib.load(MODEL_PATH)

# EXACT 18 features (use this list — do NOT change it unless retraining)
FEATURES = [
    'age', 'sex', 'on_thyroxine', 'on_antithyroid_medication', 'sick', 'pregnant',
    'thyroid_surgery', 'I131_treatment', 'lithium', 'goitre', 'tumor',
    'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'
]

# Required fields
REQUIRED_FIELDS = ['sex', 'TSH', 'T3']

# Class mapping
CLASS_MAP = {0: "Normal", 1: "Hypothyroid", 2: "Hyperthyroid"}

# Helper: encode sex to numeric expected by model
def encode_sex(value):
    if value is None:
        return 0
    v = str(value).strip().lower()
    if v in ('male', 'm', '0'):
        return 0
    if v in ('female', 'f', '1'):
        return 1
    try:
        return int(float(v))
    except:
        return 0

@app.route('/')
def home():
    # top row order as requested
    top_fields = ['TSH', 'T3', 'TT4', 'T4U', 'FTI', 'age', 'sex']
    remaining_fields = [f for f in FEATURES if f not in top_fields]
    return render_template("index.html",
                           top_fields=top_fields,
                           remaining_fields=remaining_fields,
                           required=REQUIRED_FIELDS)

@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    input_data = []
    for feature in FEATURES:
        raw = request.form.get(feature)
        # required check
        if feature in REQUIRED_FIELDS and (raw is None or str(raw).strip() == ""):
            return f"Error: Required field {feature} missing.", 400
        # default missing to 0
        if raw is None or str(raw).strip() == "":
            raw = 0
        # encode sex
        if feature == 'sex':
            val = encode_sex(raw)
        else:
            try:
                val = float(raw)
            except:
                # if user typed something non-numeric, try to extract digits
                digits = re.findall(r"[-+]?\d*\.?\d+", str(raw))
                val = float(digits[0]) if digits else 0.0
        input_data.append(float(val))

    pred_class = model.predict([input_data])[0]
    prediction = CLASS_MAP.get(int(pred_class), "Unknown")
    return render_template("result.html", prediction=prediction)

@app.route('/predict_pdf', methods=['POST'])
def predict_pdf():
    if 'pdf_file' not in request.files:
        return "No file uploaded", 400
    file = request.files['pdf_file']
    if file.filename == "":
        return "No file selected", 400

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # extract text (text-based PDFs only)
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            p = page.extract_text()
            if p:
                text += p + "\n"

    # parse features
    input_data = []
    for feature in FEATURES:
        if feature == 'sex':
            m = re.search(r"sex\s*[:=]\s*(male|female|m|f)", text, re.IGNORECASE)
            val = encode_sex(m.group(1)) if m else 0
        else:
            # look for patterns like "TSH: 3.5" or "TSH = 3.5" or "TSH 3.5"
            pattern = rf"{re.escape(feature)}\s*[:=]?\s*([-+]?\d*\.?\d+)"
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                # try a more general search using uppercase label variants (e.g., "TSH 3.5")
                pattern2 = rf"{re.escape(feature)}\b[^\d\.-]*([-+]?\d*\.?\d+)"
                match = re.search(pattern2, text, re.IGNORECASE)
            val = float(match.group(1)) if match else 0.0
        input_data.append(float(val))

    # check required fields
    for idx, feature in enumerate(FEATURES):
        if feature in REQUIRED_FIELDS and input_data[idx] == 0:
            return f"Error: Required field {feature} missing in PDF.", 400

    pred_class = model.predict([input_data])[0]
    prediction = CLASS_MAP.get(int(pred_class), "Unknown")
    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
