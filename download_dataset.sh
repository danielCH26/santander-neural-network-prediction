#!/bin/bash

echo "📥 Downloading Santander Customer Transaction Dataset..."
echo "===================================================="

# Check if file already exists
if [ -f "train.csv" ]; then
    echo "⚠️  train.csv already exists. Skipping download."
    echo "   To re-download, delete the file first:"
    echo "   rm train.csv"
    exit 0
fi

# Option 1: Download from Google Drive (recommended)
echo "📂 Attempting download from Google Drive..."
if command -v gdown &> /dev/null; then
    echo "✅ gdown found. Downloading..."
    gdown https://drive.google.com/uc?id=1Bdyw_MXfUp6BrjGcWyO096u5uiasOzGj -O train.csv

    if [ -f "train.csv" ]; then
        echo "✅ Dataset downloaded successfully!"
        echo "   File size: $(du -h train.csv | cut -f1)"
        exit 0
    else
        echo "❌ Google Drive download failed. Trying Kaggle..."
    fi
else
    echo "⚠️  gdown not found. Install with: pip install gdown"
    echo "   Trying Kaggle..."
fi

# Option 2: Download from Kaggle
echo "📂 Attempting download from Kaggle..."
if command -v kaggle &> /dev/null; then
    echo "✅ kaggle found. Downloading..."
    kaggle competitions download -c santander-customer-transaction-prediction

    if [ -f "santander-customer-transaction-prediction.zip" ]; then
        echo "✅ Download complete. Extracting..."
        unzip santander-customer-transaction-prediction.zip

        # Extract train.csv from zip
        if [ -f "sample_submission.csv.zip" ]; then
            unzip sample_submission.csv.zip
        fi
        if [ -f "test.csv.zip" ]; then
            unzip test.csv.zip
        fi

        echo "✅ Dataset extracted successfully!"
        echo "   File size: $(du -h train.csv | cut -f1)"

        # Cleanup
        rm -f *.zip sample_submission.csv test.csv
        exit 0
    else
        echo "❌ Kaggle download failed."
    fi
else
    echo "⚠️  kaggle not found. Install with: pip install kaggle"
    echo "   You also need to configure Kaggle API credentials:"
    echo "   1. Go to https://www.kaggle.com/account"
    echo "   2. Create API token (kaggle.json)"
    echo "   3. Place kaggle.json in ~/.kaggle/"
fi

echo "❌ Failed to download dataset automatically."
echo ""
echo "📚 Manual download options:"
echo ""
echo "Option 1: Google Drive (Direct download)"
echo "   URL: https://drive.google.com/file/d/1Bdyw_MXfUp6BrjGcWyO096u5uiasOzGj"
echo "   Steps: Download the file, rename it to 'train.csv', and place in this directory"
echo ""
echo "Option 2: Kaggle"
echo "   URL: https://www.kaggle.com/c/santander-customer-transaction-prediction/data"
echo "   Steps: Download 'train.csv' and place in this directory"
echo ""
echo "Dataset size: ~288 MB"
echo ""

exit 1
