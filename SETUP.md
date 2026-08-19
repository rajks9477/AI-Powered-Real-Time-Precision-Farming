# ArogyaKrishi — Setup Guide

This is a fully working Flask MVP: user signup/login, dashboard with activity
history, crop recommendation (trained RandomForest), fertilizer recommendation
(rule engine), disease detection pipeline (ResNet9 + PlantVillage-style
classes), live weather lookup, a rule-based multilingual assistant widget,
Google-Translate language switcher, and a farmer community board.

## 1. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> `torch` / `torchvision` are only needed for the image-based disease
> detector. Everything else (crop recommendation, fertilizer, auth,
> dashboard, weather, chatbot, community) works without them.

## 2. Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env`:
- `SECRET_KEY` — any random string (used to sign Flask sessions/cookies).
- `OPEN_WEATHER_APIKEY` — free key from https://openweathermap.org/api
  (needed for the weather widget on the dashboard).

## 3. The crop-recommendation model is already trained

`models/RandomForest.pkl` is included and trained on `data/Crop_recommendation.csv`
(~97% test accuracy across 22 crop classes). To retrain or extend it with your
own real-world data, replace the CSV and run:

```bash
python notebooks/train_crop_model.py
```

## 4. Enable real disease detection, soil analysis & the smart chatbot (free)

Get a **free** Hugging Face account and access token:
1. Sign up at https://huggingface.co/join
2. Go to https://huggingface.co/settings/tokens -> "Create new token" -> type **Read** -> Create.
3. Copy the token and paste it into your `.env` file:
   ```
   HUGGINGFACE_LOGIN_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
   ```
4. Restart the app (`Ctrl+C` then `python app.py`).

With this token set:
- **Disease Detection** calls a real, pre-trained public model (38 disease classes
  across 14 major crops) hosted on Hugging Face - works on any real leaf photo.
- **Chatbot** calls a free hosted LLM so it can answer open-ended farming questions,
  not just fixed keywords.
- The first request to a Hugging Face model can take 20-30 seconds while their
  server "wakes up" the model - if it times out, just try again once.

**Soil Type Detection** works immediately with no token - it's a built-in
colour/brightness heuristic (not a hosted AI model, since no reliable free
soil-type-from-photo model exists publicly). Treat its result as a helpful
starting estimate, not a lab-grade soil test.

**Live Weather** needs your own free OpenWeatherMap key (see step 2) - once
that's in `.env`, the weather widget on the dashboard will show live data.

## 4b. (Optional) Local disease-detection model instead of the API

The disease detector uses a ResNet9 CNN (`utils/model.py`) trained on
leaf-image data (PlantVillage-style, 38 classes across 14 crops).
Training a CNN needs a GPU and a large labelled image dataset, so it is
not bundled in this MVP package. To turn the feature on:

1. Get a `plant_disease_model.pth` weights file — either train your own
   (see `utils/model.py` for the exact architecture the weights must match)
   using a dataset such as the Kaggle "New Plant Diseases Dataset", or use
   the Drive link from the original project README.
2. Place it at `models/plant_disease_model.pth`.
3. `pip install torch torchvision`.

Until then, the Disease Detection page will still accept uploads but shows
a friendly notice explaining the weights are missing — every other feature
(crop, fertilizer, auth, dashboard, weather, chatbot, community) works
immediately.

## 5. Run the app

```bash
python app.py
```

Visit **http://127.0.0.1:5000** — the SQLite database
(`instance/farmers_database.db`) is created automatically on first run.

## What's included vs. what you extend

| Feature | Status |
|---|---|
| Signup / Login / Logout (hashed passwords) | ✅ working |
| Dashboard with activity history + stats | ✅ working |
| Crop recommendation (RandomForest, 22 crops) | ✅ working, trained model included |
| Fertilizer recommendation (rule engine, 22 crops) | ✅ working |
| Disease detection pipeline & UI | ✅ working, needs your trained `.pth` weights for real predictions |
| Weather widget | ✅ working, needs your free OpenWeatherMap key |
| Multilingual (Google Translate widget) | ✅ working out of the box |
| Chatbot assistant | ✅ working (offline, rule-based) |
| Community board | ✅ working |

## Project structure

```
ArogyaKrishi/
├── app.py                     # Flask app: routes, auth, DB models
├── config.py                  # App configuration
├── requirements.txt
├── .env.example
├── models/
│   └── RandomForest.pkl       # trained crop-recommendation model
├── utils/
│   ├── fertilizer.py          # NPK rule engine
│   ├── disease.py             # disease info + class labels
│   └── model.py                # ResNet9 CNN architecture + inference helper
├── notebooks/
│   ├── generate_crop_data.py  # builds data/Crop_recommendation.csv
│   └── train_crop_model.py    # trains & saves RandomForest.pkl
├── data/
│   └── Crop_recommendation.csv
├── templates/                 # all HTML pages (Jinja2)
├── static/css/style.css       # design system
├── static/js/script.js        # chatbot, upload preview, weather widget
├── uploads/                   # user-uploaded leaf images
└── instance/                  # SQLite DB (created on first run)
```
