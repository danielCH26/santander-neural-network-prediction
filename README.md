<div align="center">

  # 🧠 Deep Neural Network for Transaction Classification
  ## Santander Customer Transaction Prediction - PyTorch Production Implementation

  ![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
  ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
  ![License](https://img.shields.io/badge/License-MIT-green.svg)
  ![Production](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

  **Machine Learning Engineer Portfolio Project**
  *Advanced Classification Pipeline with PyTorch*

  ![Neural Network](https://miro.medium.com/v2/resize:fit:720/format:webp/1*1wID-4sJ7d0Yi0AvGPnP6A.png)

</div>

---

## 📊 Table of Contents

- [📝 Project Overview](#-project-overview)
- [🎯 Business Problem](#-business-problem)
- [📁 Dataset](#-dataset)
- [🏗️ Architecture](️-architecture)
- [🚀 Performance Metrics](#-performance-metrics)
- [🔧 Technical Stack](#-technical-stack)
- [💻 Installation & Usage](#-installation--usage)
- [📈 Model Deployment](#-model-deployment)
- [👤 Author](#-author)

---

## 📝 Project Overview

This project implements a **production-ready deep neural network** using **PyTorch** for binary classification in the fintech domain. The system predicts whether a customer will make a specific transaction based on **200 anonymized features**, demonstrating expertise in:

- 🧠 **Deep Learning Architecture** - Custom DNN with regularization techniques
- ⚡ **GPU Optimization** - CUDA acceleration for production-scale training
- 📊 **Feature Engineering** - High-dimensional data preprocessing pipelines
- 🎯 **Model Evaluation** - Comprehensive metrics and validation strategies
- 🔧 **Production Practices** - Early stopping, checkpointing, reproducible training

### 📌 Key Highlights

| Feature | Detail |
|---------|--------|
| **Model Type** | Deep Neural Network (200→256→128→1) |
| **Framework** | PyTorch 2.0+ with CUDA support |
| **Dataset Size** | 200,000 samples × 200 features |
| **Training Strategy** | Early stopping, Dropout, Batch Normalization |
| **Production Ready** | ✅ Checkpointing, reproducible runs |
| **GPU Acceleration** | ✅ Verified on T4 GPU (Google Colab) |

---

## 🎯 Business Problem

### Problem Statement

Predict customer transaction behavior in a banking context to enable:

- 💰 **Fraud Detection** - Identify suspicious transaction patterns
- 🎯 **Personalized Marketing** - Target customers likely to transact
- 📊 **Risk Assessment** - Evaluate customer transaction probability
- ⚡ **Resource Optimization** - Prioritize high-probability transactions

### Technical Challenge

```
┌─────────────────────────────────────────────────────────┐
│  CHALLENGE: Binary Classification on High-Dim Data     │
│                                                         │
│  200 Features (var_0 ... var_199)  ──►  Deep Neural   │
│  (Normalized, scaled)             │  Network (PyTorch) │
│                                  └──────┬────────────── │
│                                         │              │
│                              ┌──────────┴───────────┐  │
│                              ▼                      ▼  │
│                           Yes (1)                 No (0)  │
│                           (Transaction)        (No Transaction) │
└─────────────────────────────────────────────────────────┘
```

**Key Technical Challenges Addressed:**
- ✅ High-dimensional feature space (200 features)
- ✅ Class imbalance handling
- ✅ Overfitting prevention (Dropout, BatchNorm, Early Stopping)
- ✅ Scalable training pipeline (batch processing, GPU optimization)
- ✅ Reproducible results (seed setting, checkpointing)

---

## 📁 Dataset

### Santander Customer Transaction Prediction

| Characteristic | Details |
|---------------|----------|
| **Samples** | 200,000 |
| **Features** | 200 numerical (var_0 to var_199) |
| **Target** | Binary (0 = No transaction, 1 = Transaction) |
| **Problem** | Binary classification |
| **Source** | [Santander - Kaggle](https://www.kaggle.com/c/santander-customer-transaction-prediction) |
| **Data Quality** | Cleaned, preprocessed, no missing values |

### 🔧 Dataset Acquisition

**Option 1: Download from Google Drive**
```bash
# Using gdown (install with: pip install gdown)
gdown https://drive.google.com/uc?id=1Bdyw_MXfUp6BrjGcWyO096u5uiasOzGj -O train.csv
```

**Option 2: Download from Kaggle**
```bash
# Requires Kaggle API credentials
pip install kaggle
kaggle competitions download -c santander-customer-transaction-prediction
unzip santander-customer-transaction-prediction.zip
```

### 📊 Data Structure

```python
# Sample data point
{
  "ID_code": "train_0",      # Identifier (not used in training)
  "target": 0,               # Binary target (0 or 1)
  "var_0": 8.9255,           # Anonymized feature
  "var_1": -6.7863,
  "var_2": 11.9081,
  ...
  "var_199": -1.0914
}
```

### 🔄 Preprocessing Pipeline

```python
# 1. Feature selection
X = data.drop(['ID_code', 'target'], axis=1)
y = data['target']

# 2. Strategic split (70/15/15)
X_train, X_val, X_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test = train_test_split(
    X_val, X_test, test_size=0.50, stratify=y_val, random_state=42
)

# 3. StandardScaler (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. PyTorch Tensor conversion
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled),
    torch.FloatTensor(y_train.values)
)
```

---

## 🏗️ Architecture

### 🧩 SantanderNN - Custom Deep Neural Network

```
┌──────────────────────────────────────────────────────────┐
│            SANTANDER NEURAL NETWORK (SantanderNN)        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Input Layer:  200 neurons (var_0...var_199)             │
│                     ↓                                    │
│  Hidden Layer 1:                                        │
│    • 256 neurons                                        │
│    • BatchNorm1d (normalization)                        │
│    • ReLU activation                                     │
│    • Dropout(0.4) (regularization)                      │
│                     ↓                                    │
│  Hidden Layer 2:                                        │
│    • 128 neurons                                        │
│    • BatchNorm1d (normalization)                        │
│    • ReLU activation                                     │
│    • Dropout(0.3) (regularization)                      │
│                     ↓                                    │
│  Output Layer:                                           │
│    • 1 neuron (logits for binary classification)        │
│    • Sigmoid activation (via BCEWithLogitsLoss)         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### ⚙️ Training Configuration

| Component | Configuration | Rationale |
|-----------|---------------|-----------|
| **Loss Function** | `BCEWithLogitsLoss` | Combines Sigmoid + Binary Cross Entropy (numerically stable) |
| **Optimizer** | `Adam` | lr=0.001, weight_decay=1e-5 (L2 regularization) |
| **Scheduler** | ReduceLROnPlateau | Monitor val_loss, factor=0.5, patience=3 |
| **Early Stopping** | Patience=5 | Prevents overfitting, saves best model |
| **Batch Size** | 1024 | Optimal for GPU memory (tested on T4) |
| **Epochs** | Max 30 | Early stopping typically stops at 12-15 epochs |
| **Seed** | Fixed (42) | Reproducible results |

### 📊 Training Pipeline

```python
# Training loop with checkpointing
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        # Forward pass
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping (optional, for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor)
        val_loss = criterion(val_pred, y_val_tensor)

    # Early stopping + checkpointing
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break
```

---

## 🚀 Performance Metrics

### 📈 Training Progress

```
Epoch  1/30 | Train Loss: 0.3085 | Val Loss: 0.2605 | LR: 0.0010
Epoch  2/30 | Train Loss: 0.2396 | Val Loss: 0.2418 | LR: 0.0010
Epoch  3/30 | Train Loss: 0.2335 | Val Loss: 0.2445 | LR: 0.0010
Epoch  4/30 | Train Loss: 0.2280 | Val Loss: 0.2431 | LR: 0.0010
Epoch  5/30 | Train Loss: 0.2265 | Val Loss: 0.2428 | LR: 0.0010
...
Epoch 14/30 | Train Loss: 0.1981 | Val Loss: 0.2399 | LR: 0.0005
✅ Early stopping triggered at epoch 14
```

### 🎯 Evaluation Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Training Loss (Final)** | 0.1981 | Convergence achieved |
| **Validation Loss (Best)** | 0.2399 | No overfitting detected |
| **Accuracy** | See notebook | Overall classification correctness |
| **Precision (Class 1)** | See notebook | True positive rate for transactions |
| **Recall (Class 1)** | See notebook | Sensitivity to positive cases |
| **F1-Score (Class 1)** | See notebook | Balance between precision & recall |
| **AUC-ROC** | See notebook | Discriminatory power |

### 📊 Visualizations

The notebook includes comprehensive visualizations:

- **Confusion Matrix** - Heatmap of predictions vs actual
- **ROC Curve** - Sensitivity vs 1-Specificity
- **Training/Validation Loss Curves** - Learning progress over epochs
- **Class Distribution** - Target variable balance analysis

---

## 🔧 Technical Stack

### Core Dependencies

```yaml
Framework:
  - python: "^3.8"
  - torch: "^2.0.0"
  - torchvision: "^0.15.0"

Data Processing:
  - pandas: "^1.5.0"
  - numpy: "^1.23.0"
  - scikit-learn: "^1.2.0"

Visualization:
  - matplotlib: "^3.6.0"
  - seaborn: "^0.12.0"

Utilities:
  - jupyter: "^1.0.0"
  - gdown: "^4.5.0"  # For dataset download
```

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn matplotlib seaborn jupyter gdown

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 💻 Installation & Usage

### 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/danielCH26/santander-neural-network-prediction.git
cd santander-neural-network-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset
bash download_dataset.sh  # Or run the Python script manually

# 5. Open Jupyter notebook
jupyter notebook Trabajo1.ipynb
```

### 📝 Usage Examples

#### Training the Model

```python
from santander_nn import SantanderNN, train_model, evaluate_model

# Load data
data = pd.read_csv('train.csv')

# Preprocess
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(data)

# Create model
model = SantanderNN(input_dim=200)

# Train
model, history = train_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    batch_size=1024,
    epochs=30,
    early_stopping_patience=5
)

# Evaluate
metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
```

#### Inference on New Data

```python
import torch
from sklearn.preprocessing import StandardScaler
import pickle

# Load trained model
model = SantanderNN(input_dim=200)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Load scaler (saved during training)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare new data
new_data = pd.read_csv('new_transactions.csv')
X_new = scaler.transform(new_data.drop(['ID_code'], axis=1))
X_new_tensor = torch.FloatTensor(X_new)

# Make predictions
with torch.no_grad():
    predictions = model(X_new_tensor)
    probabilities = torch.sigmoid(predictions).numpy()

# Binary classification (threshold = 0.5)
predicted_classes = (probabilities >= 0.5).astype(int)
```

---

## 📈 Model Deployment

### 🐳 Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY best_model.pth .
COPY scaler.pkl .
COPY inference.py .

CMD ["python", "inference.py"]
```

### 📦 API Deployment (FastAPI)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np

app = FastAPI(title="Santander Transaction Prediction API")

class TransactionFeatures(BaseModel):
    features: List[float]

@app.post("/predict")
async def predict(data: TransactionFeatures):
    # Load model
    model = SantanderNN(input_dim=200)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Predict
    with torch.no_grad():
        features_tensor = torch.FloatTensor([data.features])
        prediction = model(features_tensor)
        probability = torch.sigmoid(prediction).item()

    return {
        "predicted_class": 1 if probability >= 0.5 else 0,
        "probability": probability,
        "confidence": abs(probability - 0.5) * 2
    }
```

---

## 👤 Author

<div align="center">

  **Daniel Elias Cordoba Howard**

  🎓 *Machine Learning Engineer* | *Data Scientist*
  🐙 *GitHub: [danielCH26](https://github.com/danielCH26)*
  🔗 *LinkedIn: [Daniel Cordoba Howard](https://www.linkedin.com/in/daniel-cordoba-howard-14472a302/)*
  📧 *danielcordobahoward@gmail.com*

  <br>

  > Passionate about building production-ready machine learning systems scalable to real-world challenges.

</div>

---

<div align="center">

  ## 🌟 Show Your Support! 🌟

  If you find this project valuable, please consider:

  - ⭐ Starring this repository on GitHub
  - 🔄 Sharing with your network
  - 💬 Providing feedback or suggestions

  **Repository:** [github.com/danielCH26/santander-neural-network-prediction](https://github.com/danielCH26/santander-neural-network-prediction)

  [⬆️ Back to Top](#-deep-neural-network-for-transaction-classification)

</div>

---

## 📄 License

MIT License - Feel free to use this project for learning and commercial purposes.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📞 Contact

For questions or collaboration opportunities, please reach out via:
- 📧 Email: danielcordobahoward@gmail.com
- 🐙 GitHub: [danielCH26](https://github.com/danielCH26)
- 💼 LinkedIn: [Daniel Cordoba Howard](https://www.linkedin.com/in/daniel-cordoba-howard-14472a302/)
