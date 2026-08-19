"""
Rule based fertilizer recommendation engine.
Compares the crop's ideal NPK requirement with the soil's current NPK value
(taken from a soil-test / sensor input) and tells the farmer whether each
nutrient is in surplus, deficit or in balance, plus which fertilizer to use.
"""

# Ideal average NPK requirement per crop (kg/ha) - agronomy reference values
crop_npk_requirement = {
    "rice": {"N": 80, "P": 40, "K": 40},
    "maize": {"N": 80, "P": 40, "K": 20},
    "chickpea": {"N": 40, "P": 60, "K": 80},
    "kidneybeans": {"N": 20, "P": 60, "K": 20},
    "pigeonpeas": {"N": 20, "P": 60, "K": 20},
    "mothbeans": {"N": 20, "P": 40, "K": 20},
    "mungbean": {"N": 20, "P": 40, "K": 20},
    "blackgram": {"N": 40, "P": 60, "K": 20},
    "lentil": {"N": 20, "P": 60, "K": 20},
    "pomegranate": {"N": 20, "P": 10, "K": 40},
    "banana": {"N": 100, "P": 75, "K": 50},
    "mango": {"N": 20, "P": 20, "K": 30},
    "grapes": {"N": 20, "P": 125, "K": 200},
    "watermelon": {"N": 100, "P": 10, "K": 50},
    "muskmelon": {"N": 100, "P": 10, "K": 50},
    "apple": {"N": 20, "P": 125, "K": 200},
    "orange": {"N": 20, "P": 10, "K": 10},
    "papaya": {"N": 50, "P": 50, "K": 50},
    "coconut": {"N": 20, "P": 10, "K": 30},
    "cotton": {"N": 120, "P": 40, "K": 20},
    "jute": {"N": 80, "P": 40, "K": 40},
    "coffee": {"N": 100, "P": 20, "K": 30},
}

fertilizer_dic = {
    "NHigh": {
        "title": "Nitrogen is high in your soil",
        "advice": (
            "Reduce nitrogen fertilizer application. Grow nitrogen consuming crops "
            "(e.g. legumes) in rotation. Add organic matter with a high C:N ratio "
            "like straw or sawdust to help microbes absorb excess nitrogen. "
            "Avoid urea; use split doses only if the crop genuinely needs it."
        ),
    },
    "Nlow": {
        "title": "Nitrogen is low in your soil",
        "advice": (
            "Apply Urea or Ammonium Sulphate. Use well-decomposed farmyard manure/compost. "
            "Grow a leguminous cover crop (e.g. cowpea, dhaincha) to fix atmospheric nitrogen. "
            "Split nitrogen dose into 2-3 applications during the crop cycle for better uptake."
        ),
    },
    "PHigh": {
        "title": "Phosphorus is high in your soil",
        "advice": (
            "Avoid phosphorus-rich fertilizers (DAP, SSP) for a season. "
            "Grow phosphorus-loving crops to draw down the excess. "
            "Excess phosphorus can lock up micronutrients like zinc and iron - monitor for deficiency symptoms."
        ),
    },
    "Plow": {
        "title": "Phosphorus is low in your soil",
        "advice": (
            "Apply Single Super Phosphate (SSP) or DAP at sowing time close to the root zone. "
            "Add well-rotted compost/bone meal. Maintain soil pH close to neutral (6.5-7) "
            "as phosphorus availability drops sharply in very acidic or alkaline soils."
        ),
    },
    "KHigh": {
        "title": "Potassium is high in your soil",
        "advice": (
            "Stop potash (MOP) application. Leach the soil with irrigation water if the field allows. "
            "Grow crops with a high potassium requirement (banana, potato) to use up the surplus."
        ),
    },
    "Klow": {
        "title": "Potassium is low in your soil",
        "advice": (
            "Apply Muriate of Potash (MOP) or Sulphate of Potash. "
            "Add wood ash or well-decomposed compost. Potassium improves disease resistance, "
            "water use efficiency and fruit/grain quality, so do not skip this correction."
        ),
    },
}


def normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "")


def recommend_fertilizer(crop_name: str, N: float, P: float, K: float):
    """
    Returns a list of {nutrient, status, title, advice} dicts, one per NPK nutrient.
    status is one of: 'high', 'low', 'balanced'
    """
    crop_key = normalize(crop_name)
    ideal = crop_npk_requirement.get(crop_key)
    if ideal is None:
        # fallback to a generic balanced requirement if crop unknown
        ideal = {"N": 60, "P": 40, "K": 40}

    results = []
    for nutrient, value in (("N", N), ("P", P), ("K", K)):
        target = ideal[nutrient]
        diff_pct = (value - target) / target * 100 if target else 0

        if diff_pct > 15:
            key = f"{nutrient}High"
            status = "high"
        elif diff_pct < -15:
            key = f"{nutrient}low"
            status = "low"
        else:
            results.append({
                "nutrient": nutrient,
                "status": "balanced",
                "title": f"{nutrient} is well balanced for {crop_name}",
                "advice": "No corrective action needed. Maintain current fertilizer schedule "
                          "and re-test soil every season.",
            })
            continue

        info = fertilizer_dic[key]
        results.append({
            "nutrient": nutrient,
            "status": status,
            "title": info["title"],
            "advice": info["advice"],
        })

    return {"crop": crop_name, "ideal": ideal, "given": {"N": N, "P": P, "K": K}, "results": results}
