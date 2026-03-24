#!/usr/bin/env python3
"""
Santander Transaction Prediction - Inference Module
===================================================

This module provides a simple interface for loading and using the trained
Santander Neural Network model for production inference.

Usage:
    python inference.py --input data.csv --output predictions.csv

Author: Daniel Elias Cordoba Howard
Email: danielcordobahoward@gmail.com
LinkedIn: https://www.linkedin.com/in/daniel-cordoba-howard-14472a302/
"""

import argparse
import pickle
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class SantanderNN(nn.Module):
    """Production-ready Santander Neural Network."""
    
    def __init__(self, input_dim=200):
        super(SantanderNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class ModelPredictor:
    """Wrapper for model inference with preprocessing."""
    
    def __init__(self, model_path: str = "best_model.pth",
                 scaler_path: Union[str, None] = None,
                 device: str = "auto"):
        """
        Initialize predictor with model and scaler.
        
        Args:
            model_path: Path to saved model weights (.pth)
            scaler_path: Path to saved scaler object (.pkl)
            device: "auto", "cuda", or "cpu"
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"📱 Device: {self.device}")
        
        # Load model
        self.model = SantanderNN(input_dim=200)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"✅ Model loaded from {model_path}")
        
        # Load scaler
        if scaler_path and Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Scaler loaded from {scaler_path}")
        else:
            print("⚠️  No scaler found. Assuming input is already scaled.")
            self.scaler = None
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame, torch.Tensor],
                threshold: float = 0.5) -> dict:
        """
        Make predictions on input data.
        
        Args:
            X: Input features (n_samples, 200)
            threshold: Classification threshold (default: 0.5)
        
        Returns:
            Dictionary with predictions and probabilities
        """
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            # Exclude ID_code if present
            if 'ID_code' in X.columns:
                X = X.drop(['ID_code'], axis=1)
            X = X.values
        
        # Apply scaler if available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Convert to tensor
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X
        
        # Move to device
        X_tensor = X_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
            classes = (probabilities >= threshold).astype(int)
        
        return {
            'probabilities': probabilities,
            'classes': classes,
            'threshold': threshold
        }
    
    def predict_file(self, input_path: str, output_path: str,
                    threshold: float = 0.5) -> None:
        """
        Load data from file, predict, and save results.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save predictions CSV
            threshold: Classification threshold
        """
        print(f"📂 Loading data from {input_path}...")
        
        # Load data
        data = pd.read_csv(input_path)
        print(f"   Shape: {data.shape}")
        
        # Preserve ID_code if exists
        id_col = None
        if 'ID_code' in data.columns:
            id_col = data['ID_code'].values
            X = data.drop(['ID_code', 'target'], axis=1, errors='ignore')
        else:
            X = data.drop(['target'], axis=1, errors='ignore')
        
        # Predict
        print("🔮 Running predictions...")
        results = self.predict(X.values, threshold=threshold)
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'prediction_class': results['classes'],
            'probability': results['probabilities']
        })
        
        # Add ID_code if exists
        if id_col is not None:
            output_df.insert(0, 'ID_code', id_col)
        
        # Save
        output_df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")
        
        # Print summary
        print("\n📊 Summary:")
        print(f"   Total samples: {len(output_df)}")
        print(f"   Predicted Class 0 (No Transaction): {np.sum(results['classes'] == 0)}")
        print(f"   Predicted Class 1 (Transaction): {np.sum(results['classes'] == 1)}")
        print(f"   Avg Probability: {results['probabilities'].mean():.4f}")


def main():
    """CLI interface for inference."""
    parser = argparse.ArgumentParser(
        description="Santander Transaction Prediction - Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Predict on CSV file
    python inference.py --input test_data.csv --output predictions.csv
    
    # Predict on CSV file with custom threshold
    python inference.py --input test_data.csv --output predictions.csv --threshold 0.6
    
    # Use CPU only
    python inference.py --input test_data.csv --output predictions.csv --device cpu

Requirements:
    - Input CSV must have 200 feature columns (var_0 to var_199)
    - Model file: best_model.pth (must be in current directory)
    - Optional: scaler.pkl (for preprocessing, must be in current directory)
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default='predictions.csv',
                       help='Output CSV file path (default: predictions.csv)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--model', type=str, default='best_model.pth',
                       help='Model weights path (default: best_model.pth)')
    parser.add_argument('--scaler', type=str, default='scaler.pkl',
                       help='Scaler path (default: scaler.pkl)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device for inference (default: auto)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        print("   Train the model first: jupyter notebook Trabajo1.ipynb")
        return 1
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Input file not found: {args.input}")
        return 1
    
    # Initialize predictor
    predictor = ModelPredictor(
        model_path=args.model,
        scaler_path=args.scaler,
        device=args.device
    )
    
    # Run prediction
    predictor.predict_file(
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
