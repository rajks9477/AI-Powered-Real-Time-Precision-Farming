import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "instance", "farmers_database.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

    OPEN_WEATHER_APIKEY = os.environ.get("OPEN_WEATHER_APIKEY", "")
    HUGGINGFACE_LOGIN_TOKEN = os.environ.get("HUGGINGFACE_LOGIN_TOKEN", "")

    CROP_MODEL_PATH = os.path.join(BASE_DIR, "models", "RandomForest.pkl")
    DISEASE_MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease_model.pth")