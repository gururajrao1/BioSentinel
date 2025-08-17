# BioSentinel

**BioSentinel** is a machine learning pipeline for trauma and battlefield wound detection. Using limited medical imaging datasets, it classifies and visualizes wounds with around **74% accuracy** using a **ResNet-18** backbone. The project demonstrates end-to-end workflow: from data preprocessing to model training and evaluation, highlighting challenges of small datasets in medical AI.

## Features
- End-to-end ML pipeline for trauma wound detection
- Data preprocessing and augmentation
- ResNet-18 model for classification
- Visualization of wounds and predictions
- Demonstrates working with limited datasets

## Dataset
The model is trained on **limited medical imaging datasets** (not included in the repo due to size and privacy). You can provide your own datasets in the same format for experimentation. 

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/BioSentinel.git
Create a virtual environment and install dependencies:
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt

Notes

Accuracy is around 74% due to dataset scarcity.

Jupyter notebooks are included for visualizing workflow steps.

Do not commit large dataset files to this repo. Use .gitignore.


