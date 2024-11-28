import os
from tensorflow.keras.models import load_model # type: ignore
import joblib
import pandas as pd # type: ignore
from django.conf import settings





def income_model(data):
    model_path = os.path.join(settings.BASE_DIR, 'models', 'customer_income_model.h5')
    scaler_path = os.path.join(settings.BASE_DIR, 'models', 'customer_income_scaler.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    scaled_data = scaler.transform(data)

    # 예측
    predictions = model.predict(scaled_data)

    return predictions