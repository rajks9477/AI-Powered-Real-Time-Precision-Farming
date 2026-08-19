"""
Generates a realistic Crop Recommendation dataset (N,P,K,temperature,humidity,ph,rainfall -> label)
Ranges are based on published agronomy references (ICAR / Kaggle crop-recommendation dataset schema)
so the trained model behaves sensibly for the 22 crop classes used in the app.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# crop: (N range, P range, K range, temp range, humidity range, ph range, rainfall range)
CROPS = {
    "rice":        ((60,100),(30,60),(30,50),(20,32),(75,95),(5.5,7.5),(150,300)),
    "maize":       ((60,100),(35,65),(15,35),(18,27),(50,75),(5.5,7.5),(60,120)),
    "chickpea":    ((20,50),(55,85),(70,100),(15,25),(15,35),(6.0,8.0),(60,100)),
    "kidneybeans": ((15,40),(55,85),(15,30),(15,25),(15,35),(5.5,6.5),(60,120)),
    "pigeonpeas":  ((15,40),(55,85),(15,30),(18,32),(30,65),(4.5,7.5),(90,200)),
    "mothbeans":   ((15,40),(35,65),(15,30),(24,32),(30,65),(3.5,10.0),(30,70)),
    "mungbean":    ((15,40),(35,65),(15,30),(25,35),(60,90),(6.0,7.5),(40,90)),
    "blackgram":   ((15,40),(55,85),(15,30),(25,35),(60,90),(6.0,7.5),(60,90)),
    "lentil":      ((15,40),(55,85),(15,30),(15,25),(60,90),(5.5,7.5),(35,80)),
    "pomegranate": ((10,30),(10,30),(30,50),(18,25),(85,95),(6.0,7.5),(100,120)),
    "banana":      ((80,120),(70,100),(45,65),(22,32),(75,90),(5.5,6.5),(100,180)),
    "mango":       ((10,30),(15,35),(25,45),(24,35),(45,65),(5.5,7.5),(80,120)),
    "grapes":      ((10,30),(120,150),(190,210),(15,35),(75,85),(5.5,6.5),(60,90)),
    "watermelon":  ((80,110),(10,30),(45,65),(24,32),(55,75),(6.0,7.0),(40,70)),
    "muskmelon":   ((80,110),(10,30),(45,65),(25,32),(85,95),(6.0,7.0),(20,30)),
    "apple":       ((15,40),(120,150),(190,210),(15,25),(85,95),(5.5,6.5),(100,130)),
    "orange":      ((10,30),(5,25),(5,25),(15,32),(85,95),(6.0,8.0),(100,130)),
    "papaya":      ((30,70),(45,75),(45,65),(22,35),(85,95),(6.0,7.0),(40,120)),
    "coconut":     ((10,30),(5,30),(25,45),(24,32),(90,100),(5.5,7.5),(120,220)),
    "cotton":      ((90,120),(35,65),(15,35),(22,32),(65,85),(5.5,8.0),(60,110)),
    "jute":        ((60,100),(35,60),(30,50),(23,32),(70,90),(6.0,7.5),(150,200)),
    "coffee":      ((80,120),(15,35),(25,45),(18,28),(50,70),(6.0,7.5),(150,220)),
}

rows = []
for crop, (N,P,K,T,H,PH,R) in CROPS.items():
    for _ in range(150):
        rows.append({
            "N": round(np.random.uniform(*N),1),
            "P": round(np.random.uniform(*P),1),
            "K": round(np.random.uniform(*K),1),
            "temperature": round(np.random.uniform(*T),2),
            "humidity": round(np.random.uniform(*H),2),
            "ph": round(np.random.uniform(*PH),2),
            "rainfall": round(np.random.uniform(*R),2),
            "label": crop
        })

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("/home/claude/ArogyaKrishi/data/Crop_recommendation.csv", index=False)
print(df.shape)
print(df.head())
