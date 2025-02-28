# Electricity Theft Detection using Deep Learning

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction
Electricity theft is a serious issue that results in billions of dollars in losses worldwide. This project employs deep learning techniques to detect electricity theft cyber-attacks in smart grid systems. The models used include:
- Deep Feed Forward Neural Network (DNN)
- Recurrent Neural Network with Gated Recurrent Unit (RNN-GRU)
- Convolutional Neural Network (CNN)

## Project Overview
The project implements various deep learning models to classify electricity consumption data and detect anomalies indicative of electricity theft. The CNN model has shown the highest accuracy in identifying fraudulent activities.

## Dataset
The dataset used in this project is sourced from Kaggle:
[Electricity Theft Detection Dataset](https://www.kaggle.com/mrmorj/fraud-detection-in-electricity-and-gas-consumption?select=client_train.csv)

It includes:
- Smart meter readings
- Customer information
- Labels: `0` (No Theft) and `1` (Theft)

## Project Structure
```
├── ElectricityTheftDetection.py  # Main script for training and prediction
├── cnn_model.json                # CNN model architecture
├── cnn_model_weights.h5          # CNN model weights
├── gru_model.json                # GRU model architecture
├── gru_model_weights.h5          # GRU model weights
├── model.json                    # DNN model architecture
├── model_weights.h5              # DNN model weights
├── ElectricityTheft.csv          # Training dataset
├── test.csv                      # Test dataset
├── datasetlink.txt               # Link to dataset source
├── SCREENS.docx                  # Screenshots of the application
├── A14. Electricity Theft docx.docx  # Detailed project documentation
├── run.bat                       # Batch file to run the application
└── README.md                     # Project documentation
```

## Installation
### Requirements
Ensure you have the following dependencies installed:
```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib
```

### Running the Project
1. Clone the repository:
```bash
git clone https://github.com/your-username/electricity-theft-detection.git
cd electricity-theft-detection
```
2. Run the Python script:
```bash
python ElectricityTheftDetection.py
```
3. For the GUI version, double-click `run.bat`.

## Usage
- **Upload Dataset**: Load the electricity theft dataset.
- **Preprocess Dataset**: Cleans and transforms data for model training.
- **Train Models**:
  - DNN: Deep Feed Forward Neural Network.
  - RNN-GRU: Recurrent Neural Network.
  - CNN: Convolutional Neural Network.
- **Predict Electricity Theft**: Upload new data and classify fraudulent activities.
- **Comparison Graph**: Generates performance metrics for all models.

## Results
- **CNN achieved 95.98% accuracy.**
- **DNN achieved 94.24% accuracy.**
- **GRU achieved 40.02% accuracy.**

## Future Improvements
- Integration with IoT-based real-time smart meter monitoring.
- Implementing advanced feature engineering for better fraud detection.
- Using ensemble learning techniques to improve model performance.

## Acknowledgments
- **Kaggle** for the dataset.
- **Deep learning community** for inspiration and methodologies.
- **Contributors** for their efforts in enhancing electricity theft detection.

