"""
Disease information dictionary + label list for the plant-disease classifier.
The label ordering MUST match the order the model was trained on (PlantVillage-style,
38 classes: 14 crops x healthy/diseased variants).
"""

disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

disease_dic = {
    'Apple___Apple_scab': {
        'cause': 'Fungus Venturia inaequalis, spreads in cool wet spring weather.',
        'symptoms': 'Olive-green to black velvety spots on leaves and fruit; leaves may curl and drop early.',
        'treatment': 'Apply a protectant fungicide (captan/mancozeb) from bud break at 7-10 day intervals; remove fallen leaves in autumn to cut the fungus source.',
    },
    'Apple___Black_rot': {
        'cause': 'Fungus Botryosphaeria obtusa infecting through wounds and dead wood.',
        'symptoms': 'Purple leaf spots, brown rotting fruit with concentric rings, sunken bark cankers.',
        'treatment': 'Prune out dead/cankered wood, remove mummified fruit, apply fungicide sprays during the growing season.',
    },
    'Apple___Cedar_apple_rust': {
        'cause': 'Fungus Gymnosporangium juniperi-virginianae, needs a nearby juniper/cedar host.',
        'symptoms': 'Bright orange-yellow spots on leaves, small growths on fruit.',
        'treatment': 'Remove nearby juniper hosts if possible, apply fungicide at pink-bud to petal-fall stage.',
    },
    'Corn_(maize)___Common_rust_': {
        'cause': 'Fungus Puccinia sorghi, favoured by cool moist weather.',
        'symptoms': 'Small cinnamon-brown pustules on both leaf surfaces.',
        'treatment': 'Plant resistant hybrids, apply foliar fungicide if rust appears before tasseling.',
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'cause': 'Fungus Exserohilum turcicum, spreads via wind and rain-splash from crop debris.',
        'symptoms': 'Long cigar-shaped grey-green lesions on leaves.',
        'treatment': 'Rotate crops, till under residue, use resistant hybrids, apply fungicide if severe.',
    },
    'Grape___Black_rot': {
        'cause': 'Fungus Guignardia bidwellii.',
        'symptoms': 'Reddish-brown circular leaf spots and shrivelled black "mummified" berries.',
        'treatment': 'Remove mummified berries and infected canes, apply fungicide from early shoot growth through veraison.',
    },
    'Potato___Early_blight': {
        'cause': 'Fungus Alternaria solani.',
        'symptoms': 'Dark concentric-ring "target" spots on older leaves first.',
        'treatment': 'Use certified disease-free seed, rotate crops, apply chlorothalonil/mancozeb fungicide, avoid overhead irrigation.',
    },
    'Potato___Late_blight': {
        'cause': 'Oomycete Phytophthora infestans, spreads fast in cool wet weather (the Irish famine pathogen).',
        'symptoms': 'Water-soaked dark lesions on leaves with white fungal growth on the underside; can destroy a field within days.',
        'treatment': 'Destroy infected plants immediately, apply protectant fungicide preventively during humid weather, avoid overhead watering.',
    },
    'Tomato___Bacterial_spot': {
        'cause': 'Bacteria Xanthomonas spp., spreads via splashing water and contaminated tools.',
        'symptoms': 'Small dark greasy spots on leaves and fruit with yellow halo.',
        'treatment': 'Use copper-based bactericide, avoid working with wet plants, use disease-free seed/transplants.',
    },
    'Tomato___Early_blight': {
        'cause': 'Fungus Alternaria solani.',
        'symptoms': 'Concentric target-like brown spots starting on lower/older leaves.',
        'treatment': 'Remove lower infected leaves, mulch to prevent soil splash, apply fungicide, rotate crops.',
    },
    'Tomato___Late_blight': {
        'cause': 'Oomycete Phytophthora infestans.',
        'symptoms': 'Large irregular water-soaked patches turning brown, white mould on leaf undersides in humid weather.',
        'treatment': 'Remove and destroy infected plants, apply fungicide preventively, ensure good field drainage and airflow.',
    },
    'Tomato___Leaf_Mold': {
        'cause': 'Fungus Passalora fulva, favoured by high humidity.',
        'symptoms': 'Pale green/yellow spots on top of leaf, olive-green mould underneath.',
        'treatment': 'Improve ventilation, reduce humidity, apply fungicide, use resistant varieties.',
    },
    'Tomato___Septoria_leaf_spot': {
        'cause': 'Fungus Septoria lycopersici.',
        'symptoms': 'Small circular spots with dark border and grey centre, mainly on older leaves.',
        'treatment': 'Remove infected leaves, mulch, avoid overhead watering, apply fungicide.',
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'cause': 'Tetranychus urticae mite infestation, worse in hot dry weather.',
        'symptoms': 'Tiny yellow speckles/stippling on leaves, fine webbing, leaves turn bronze.',
        'treatment': 'Spray with water to dislodge mites, use insecticidal soap/neem oil, introduce predatory mites.',
    },
    'Tomato___Target_Spot': {
        'cause': 'Fungus Corynespora cassiicola.',
        'symptoms': 'Brown concentric-ring lesions on leaves, stems and fruit.',
        'treatment': 'Improve air circulation, remove infected debris, apply fungicide.',
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'cause': 'Virus transmitted by whiteflies.',
        'symptoms': 'Upward curling, yellowing, stunted growth of the whole plant.',
        'treatment': 'Control whitefly population, use resistant varieties, remove and destroy infected plants.',
    },
    'Tomato___Tomato_mosaic_virus': {
        'cause': 'Virus spread by contact, contaminated tools and hands.',
        'symptoms': 'Mottled light/dark green mosaic pattern on leaves, stunted growth.',
        'treatment': 'Remove infected plants, disinfect tools, wash hands between handling plants, use resistant varieties.',
    },
}


def normalize_label(raw_label: str) -> str:
    """
    Different sources (local model vs Hugging Face hosted model) may format
    labels slightly differently (spaces vs underscores, casing). This maps
    whatever comes in to the closest entry in `disease_classes`.
    """
    if raw_label in disease_classes:
        return raw_label
    cleaned = raw_label.strip().replace(" ", "_")
    for cls in disease_classes:
        if cls.lower() == cleaned.lower():
            return cls
    # loose match: same crop + same first keyword
    cleaned_lower = cleaned.lower()
    for cls in disease_classes:
        if cls.lower().replace("_", "") in cleaned_lower.replace("_", "") or \
           cleaned_lower.replace("_", "") in cls.lower().replace("_", ""):
            return cls
    return raw_label


def get_disease_info(disease_key: str):
    """Look up disease info; returns a generic fallback if not in dictionary (e.g. *_healthy classes)."""
    disease_key = normalize_label(disease_key)
    if disease_key.endswith('healthy'):
        crop = disease_key.split('___')[0].replace('_', ' ')
        return {
            'cause': 'N/A',
            'symptoms': f'No disease detected - the {crop} leaf appears healthy.',
            'treatment': 'Continue regular monitoring, balanced fertilization and good field hygiene.',
        }
    return disease_dic.get(disease_key, {
        'cause': 'Not in the local knowledge base yet.',
        'symptoms': 'Please consult your local agricultural extension officer for confirmation.',
        'treatment': 'General advice: remove and destroy affected leaves, avoid overhead irrigation, rotate crops next season.',
    })
