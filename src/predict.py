"""
SpaceX Falcon 9 Booster Landing - Inference Script
===================================================
Load the trained model and make predictions on new data.

Usage:
    python predict.py --rocket_name "Falcon 9" --payload_mass 5000 --orbit "ISS"
"""

import joblib
import pandas as pd
import argparse
import json
from pathlib import Path

def predict_landing_success(rocket_name, payload_mass, orbit, model_path='../src/models/best_model.pkl'):
    """
    Predict whether a SpaceX booster will successfully land.
    
    Args:
        rocket_name: Name of the rocket (e.g., "Falcon 9", "Falcon Heavy")
        payload_mass: Payload mass in kg
        orbit: Target orbit (e.g., "ISS", "GTO", "LEO")
        model_path: Path to the saved model
    
    Returns:
        dict with prediction and probability
    """
    # Load model
    model = joblib.load(model_path)
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'rocket_name': [rocket_name],
        'payload_mass': [payload_mass],
        'orbit': [orbit]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    result = {
        'prediction': 'Success' if prediction == 1 else 'Failure',
        'probability': round(float(probability), 4),
        'confidence': f"{max(probability, 1-probability)*100:.2f}%"
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Predict SpaceX booster landing success')
    parser.add_argument('--rocket_name', type=str, required=True, help='Rocket name (e.g., "Falcon 9")')
    parser.add_argument('--payload_mass', type=float, required=True, help='Payload mass in kg')
    parser.add_argument('--orbit', type=str, required=True, help='Target orbit (e.g., "ISS", "GTO")')
    
    args = parser.parse_args()
    
    result = predict_landing_success(args.rocket_name, args.payload_mass, args.orbit)
    
    print("\n" + "="*50)
    print("PREDICTION RESULT")
    print("="*50)
    print(f"Rocket:        {args.rocket_name}")
    print(f"Payload Mass:  {args.payload_mass} kg")
    print(f"Orbit:         {args.orbit}")
    print("-" * 50)
    print(f"Prediction:    {result['prediction']}")
    print(f"Probability:   {result['probability']:.4f}")
    print(f"Confidence:    {result['confidence']}")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()
