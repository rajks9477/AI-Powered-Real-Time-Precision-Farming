import os
import io
import base64
import requests
import joblib
import numpy as np
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user,
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from config import Config
from utils.fertilizer import recommend_fertilizer
from utils.disease import disease_classes, get_disease_info, normalize_label
from utils.hf_client import diagnose_leaf_image, ask_chatbot
from utils.soil import estimate_soil_type

# ----------------------------------------------------------------------------
# App setup
# ----------------------------------------------------------------------------
app = Flask(__name__)
app.config.from_object(Config)
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "instance"), exist_ok=True)

db = SQLAlchemy(app)

# FIX: Yeh line database tables ko production (Render) par start hote hi create kar degi
with app.app_context():
    db.create_all()

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."


# ----------------------------------------------------------------------------
# Database models
# ----------------------------------------------------------------------------
class Farmer(UserMixin, db.Model):
    __tablename__ = "farmers"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    location = db.Column(db.String(120))
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class ActivityLog(db.Model):
    """Tracks every crop / fertilizer / disease query a farmer makes, for the dashboard."""
    __tablename__ = "activity_logs"
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("farmers.id"), nullable=False)
    activity_type = db.Column(db.String(50))   # crop | fertilizer | disease
    input_summary = db.Column(db.String(255))
    result_summary = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class CommunityPost(db.Model):
    """Simple community knowledge-sharing board."""
    __tablename__ = "community_posts"
    id = db.Column(db.Integer, primary_key=True)
    farmer_id = db.Column(db.Integer, db.ForeignKey("farmers.id"), nullable=False)
    farmer_name = db.Column(db.String(120))
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return Farmer.query.get(int(user_id))


# ----------------------------------------------------------------------------
# Load ML models once at startup
# ----------------------------------------------------------------------------
crop_model = None
if os.path.exists(app.config["CROP_MODEL_PATH"]):
    crop_model = joblib.load(app.config["CROP_MODEL_PATH"])

disease_model = None
try:
    from utils.model import load_model as load_disease_model, predict_image
    disease_model = load_disease_model(app.config["DISEASE_MODEL_PATH"], num_classes=len(disease_classes))
except ImportError:
    # torch / torchvision not installed - disease image prediction will show a friendly notice
    predict_image = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def log_activity(activity_type, input_summary, result_summary):
    if current_user.is_authenticated:
        db.session.add(ActivityLog(
            farmer_id=current_user.id,
            activity_type=activity_type,
            input_summary=input_summary,
            result_summary=result_summary,
        ))
        db.session.commit()


# ----------------------------------------------------------------------------
# Public routes
# ----------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        location = request.form.get("location", "").strip()
        password = request.form.get("password", "")

        if not (name and email and password):
            flash("Name, email and password are required.", "error")
            return redirect(url_for("signup"))

        if Farmer.query.filter_by(email=email).first():
            flash("An account with this email already exists. Please log in.", "error")
            return redirect(url_for("login"))

        farmer = Farmer(name=name, email=email, phone=phone, location=location)
        farmer.set_password(password)
        db.session.add(farmer)
        db.session.commit()

        login_user(farmer)
        flash(f"Welcome to ArogyaKrishi, {name}!", "success")
        return redirect(url_for("dashboard"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        farmer = Farmer.query.filter_by(email=email).first()
        if farmer and farmer.check_password(password):
            login_user(farmer)
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "error")
        return redirect(url_for("login"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))


# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    logs = (ActivityLog.query
            .filter_by(farmer_id=current_user.id)
            .order_by(ActivityLog.created_at.desc())
            .limit(10).all())
    stats = {
        "crop_queries": ActivityLog.query.filter_by(farmer_id=current_user.id, activity_type="crop").count(),
        "fertilizer_queries": ActivityLog.query.filter_by(farmer_id=current_user.id, activity_type="fertilizer").count(),
        "disease_queries": ActivityLog.query.filter_by(farmer_id=current_user.id, activity_type="disease").count(),
        "soil_queries": ActivityLog.query.filter_by(farmer_id=current_user.id, activity_type="soil").count(),
    }
    return render_template("dashboard.html", logs=logs, stats=stats)


# ----------------------------------------------------------------------------
# Crop recommendation
# ----------------------------------------------------------------------------
@app.route("/crop", methods=["GET", "POST"])
def crop():
    if request.method == "POST":
        try:
            N = float(request.form["nitrogen"])
            P = float(request.form["phosphorus"])
            K = float(request.form["potassium"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])
        except (KeyError, ValueError):
            flash("Please fill all fields with valid numbers.", "error")
            return redirect(url_for("crop"))

        if crop_model is None:
            return render_template("try_again.html",
                                   message="Crop recommendation model not found. Run notebooks/train_crop_model.py first.")

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        prediction = crop_model.predict(features)[0]
        probabilities = crop_model.predict_proba(features)[0]
        confidence = round(max(probabilities) * 100, 2)

        top3_idx = np.argsort(probabilities)[-3:][::-1]
        top3 = [(crop_model.classes_[i], round(probabilities[i] * 100, 2)) for i in top3_idx]

        log_activity("crop", f"N{N} P{P} K{K} T{temperature} H{humidity} pH{ph} R{rainfall}",
                     f"{prediction} ({confidence}%)")

        return render_template("crop-result.html", prediction=prediction,
                               confidence=confidence, top3=top3)

    return render_template("crop.html")


# ----------------------------------------------------------------------------
# Fertilizer recommendation
# ----------------------------------------------------------------------------
@app.route("/fertilizer", methods=["GET", "POST"])
def fertilizer():
    if request.method == "POST":
        crop_name = request.form.get("crop_name", "").strip()
        try:
            N = float(request.form["nitrogen"])
            P = float(request.form["phosphorus"])
            K = float(request.form["potassium"])
        except (KeyError, ValueError):
            flash("Please fill all fields with valid numbers.", "error")
            return redirect(url_for("fertilizer"))

        result = recommend_fertilizer(crop_name, N, P, K)
        log_activity("fertilizer", f"{crop_name} N{N} P{P} K{K}",
                     ", ".join(r["status"] for r in result["results"]))

        return render_template("fertilizer.html", result=result, show_result=True)

    return render_template("fertilizer.html", result=None, show_result=False)


# ----------------------------------------------------------------------------
# Disease detection
# ----------------------------------------------------------------------------
@app.route("/disease", methods=["GET", "POST"])
def disease():
    if request.method == "GET":
        return render_template("disease.html")

    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Please upload a leaf image.", "error")
        return redirect(url_for("disease"))

    if not allowed_file(file.filename):
        flash("Only PNG/JPG/JPEG images are supported.", "error")
        return redirect(url_for("disease"))

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image_bytes = file.read()
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    hf_token = app.config["HUGGINGFACE_LOGIN_TOKEN"]
    ai_result = diagnose_leaf_image(image_bytes, hf_token)

    if ai_result:
        crop_name = str(ai_result.get("crop", "Unknown")).strip()
        is_healthy = bool(ai_result.get("healthy"))
        disease_name = "" if is_healthy else str(ai_result.get("disease", "")).strip()
        confidence = ai_result.get("confidence", 80)
        info = {
            "cause": ai_result.get("cause", "N/A") if not is_healthy else "N/A",
            "symptoms": ai_result.get("symptoms", "No disease detected - leaf appears healthy.") if not is_healthy
                        else "No disease detected - the leaf appears healthy.",
            "treatment": ai_result.get("treatment", "Continue regular monitoring and good field hygiene."),
        }
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        log_activity("disease", filename, f"{crop_name} - {disease_name or 'healthy'} ({confidence}%)")

        return render_template(
            "disease-result.html",
            crop_name=crop_name,
            disease_name=disease_name,
            confidence=confidence,
            info=info,
            image_b64=image_b64,
        )

    if disease_model is not None and predict_image is not None:
        label, confidence = predict_image(image_bytes, disease_model, disease_classes)
        label = normalize_label(label)
        info = get_disease_info(label)
        crop_name, disease_name = (label.split("___") + [""])[:2]
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        log_activity("disease", filename, f"{label} ({confidence}%)")
        return render_template(
            "disease-result.html",
            crop_name=crop_name.replace("_", " "),
            disease_name=disease_name.replace("_", " "),
            confidence=confidence,
            info=info,
            image_b64=image_b64,
        )

    return render_template(
        "try_again.html",
        message=("Couldn't reach the disease detection service. Double-check that "
                 "HUGGINGFACE_LOGIN_TOKEN in your .env is a valid token with "
                 "'Make calls to Inference Providers' permission, and that you have an internet "
                 "connection."),
    )


# ----------------------------------------------------------------------------
# Soil type detection
# ----------------------------------------------------------------------------
@app.route("/soil", methods=["GET", "POST"])
def soil():
    if request.method == "GET":
        return render_template("soil.html")

    file = request.files.get("image")
    if not file or file.filename == "":
        flash("Please upload a soil photo.", "error")
        return redirect(url_for("soil"))

    if not allowed_file(file.filename):
        flash("Only PNG/JPG/JPEG images are supported.", "error")
        return redirect(url_for("soil"))

    image_bytes = file.read()
    try:
        profile = estimate_soil_type(image_bytes)
    except Exception:
        flash("Couldn't read that image, please try another photo.", "error")
        return redirect(url_for("soil"))

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    log_activity("soil", file.filename, profile["label"])

    return render_template("soil-result.html", profile=profile, image_b64=image_b64)


# ----------------------------------------------------------------------------
# Weather
# ----------------------------------------------------------------------------
@app.route("/api/weather")
def api_weather():
    city = request.args.get("city", "").strip()
    if not city:
        return jsonify({"error": "city query param required"}), 400

    api_key = app.config["OPEN_WEATHER_APIKEY"]
    if not api_key:
        return jsonify({"error": "OPEN_WEATHER_APIKEY not configured in .env"}), 500

    try:
        resp = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=8,
        )
        data = resp.json()
        if resp.status_code != 200:
            return jsonify({"error": data.get("message", "weather lookup failed")}), resp.status_code

        return jsonify({
            "city": data.get("name"),
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
        })
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 502


# ----------------------------------------------------------------------------
# Chatbot
# ----------------------------------------------------------------------------
CHATBOT_RULES = [
    (["disease", "bimari", "rog"], "Go to the 'Disease Detection' page and upload a clear photo of the affected leaf - I'll identify the disease and suggest treatment."),
    (["crop", "fasal", "which crop", "kaunsi fasal"], "Visit the 'Crop Recommendation' page and enter your soil NPK values, temperature, humidity, pH and rainfall to get the best crop suggestion."),
    (["fertilizer", "khaad", "urvarak"], "Open the 'Fertilizer Recommendation' page, select your crop and enter your soil test NPK values to get a tailored fertilizer plan."),
    (["weather", "mausam"], "Check the weather widget on your dashboard, or ask me 'weather in <your city>'."),
    (["hello", "hi", "namaste"], "Namaste! I'm the ArogyaKrishi assistant. Ask me about crop disease, crop recommendation, fertilizer or weather."),
]


@app.route("/api/chatbot", methods=["POST"])
def api_chatbot():
    raw_message = (request.json or {}).get("message", "")
    message = raw_message.lower()

    hf_token = app.config["HUGGINGFACE_LOGIN_TOKEN"]
    ai_reply = ask_chatbot(raw_message, hf_token) if raw_message.strip() else None
    if ai_reply:
        return jsonify({"reply": ai_reply, "source": "ai"})

    for keywords, reply in CHATBOT_RULES:
        if any(k in message for k in keywords):
            return jsonify({"reply": reply, "source": "rule"})

    return jsonify({
        "reply": ("I can help with crop disease detection, crop recommendation, fertilizer advice, "
                  "soil type and weather."),
        "source": "rule",
    })


# ----------------------------------------------------------------------------
# Community
# ----------------------------------------------------------------------------
@app.route("/community", methods=["GET", "POST"])
@login_required
def community():
    if request.method == "POST":
        content = request.form.get("content", "").strip()
        if content:
            db.session.add(CommunityPost(farmer_id=current_user.id, farmer_name=current_user.name, content=content))
            db.session.commit()
        return redirect(url_for("community"))

    posts = CommunityPost.query.order_by(CommunityPost.created_at.desc()).limit(50).all()
    return render_template("community.html", posts=posts)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)