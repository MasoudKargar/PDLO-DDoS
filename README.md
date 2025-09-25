# DDoS Detection with Counter-Based Sampling and CNN

This repository contains a two-part project for DDoS attack detection combining data preprocessing through counter-based sampling and deep learning classification using a CNN model.

## Project Overview

The project consists of two main components:

1. **Counter-Based Sampling**: A preprocessing tool that intelligently samples network packets from PCAP files to create balanced datasets
2. **CNN DDoS Detection**: A lightweight CNN-based deep learning solution for DDoS attack detection (based on LUCID framework)

## Project Structure

```
├── Counter-Based Sampling/          # Part 1: Data preprocessing
│   └── Counter-Based Sampling.py   # Main sampling algorithm
├── CNN_DDoS_detection/             # Part 2: CNN-based detection
│   ├── lucid_cnn.py               # Main CNN model implementation
│   ├── lucid_dataset_parser.py    # Dataset parsing utilities
│   ├── util_functions.py          # Utility functions
│   ├── README.md                  # Original LUCID documentation
│   ├── sample-dataset/            # Sample data directory
│   └── output/                    # Model outputs and results
├── README.md                      # This file
├── requirements.txt               # Python dependencies
└── .gitignore                    # Git ignore rules
```

## Part 1: Counter-Based Sampling

### Overview

The Counter-Based Sampling component provides an intelligent method to reduce the size of large PCAP files while maintaining the statistical properties and balance of the original dataset. This is crucial for creating manageable datasets for machine learning training.

### Features

- **Balanced Sampling**: Maintains representation of different traffic patterns
- **Flow-based Grouping**: Groups packets by (source IP, destination IP, protocol) tuples
- **Configurable Ratios**: Supports multiple sampling ratios (5% to 95%)
- **Minimum Guarantees**: Ensures minimum representation per traffic pattern

### Usage

```python
# Example usage
from Counter-Based Sampling import reduce_pcap_balanced, process_folder

# Process a single PCAP file
reduce_pcap_balanced('input.pcap', 'output.pcap', keep_ratio=0.1, min_per_key=100)

# Process all PCAP files in a folder
process_folder('./pcap_files/')
```

### Algorithm Details

1. **Key Extraction**: Extracts (src_ip, dst_ip, protocol) from each packet
2. **Pattern Counting**: Groups packets by their keys and counts occurrences
3. **Balanced Sampling**: Samples from each group proportionally
4. **Gap Filling**: Adds additional packets if needed to reach target ratio

## Part 2: CNN DDoS Detection (LUCID-based)

### Overview

This component implements a lightweight Convolutional Neural Network for DDoS attack detection. It's based on the LUCID framework but has been adapted for new datasets and includes additional features.

### Features

- **Lightweight CNN Architecture**: Optimized for resource-constrained environments
- **Multiple Dataset Support**: Compatible with various DDoS datasets
- **Performance Metrics**: Comprehensive evaluation including FLOPs calculation
- **Hyperparameter Tuning**: Grid search and randomized search capabilities
- **Visualization**: Training progress and performance plots

### Key Components

- `lucid_cnn.py`: Main CNN model implementation and training
- `lucid_dataset_parser.py`: Dataset preprocessing and parsing utilities
- `util_functions.py`: Helper functions for data processing and evaluation

### Model Architecture

The CNN model includes:

- Convolutional layers for feature extraction
- Global max pooling for dimensionality reduction
- Dense layers for classification
- Dropout for regularization

## Installation

### Prerequisites

- Python 3.9 or higher
- NVIDIA GPU (optional but recommended for faster training)

### Environment Setup

1. Clone this repository:

```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Additional Requirements

For PCAP processing, you may need to install additional system dependencies:

- **Linux/Mac**: `sudo apt-get install tshark` or `brew install wireshark`
- **Windows**: Install Wireshark from the official website

## Usage Guide

### Step 1: Data Preprocessing (Counter-Based Sampling)

```python
# Navigate to Counter-Based Sampling directory
cd "Counter-Based Sampling"

# Modify the folder path in the script
# Edit Counter-Based Sampling.py and change:
# folder_path = "./XXX"  # Replace with your PCAP folder path

python "Counter-Based Sampling.py"
```

### Step 2: Dataset Preparation for CNN

```python
# Navigate to CNN directory
cd CNN_DDoS_detection

# Parse your dataset (first step)
python lucid_dataset_parser.py --dataset_type CUSTOM --dataset_folder ./your-data/ --packets_per_flow 10 --dataset_id YOUR_DATASET --traffic_type all --time_window 10

# Preprocess the parsed data (second step)
python lucid_dataset_parser.py --preprocess_folder ./your-data/
```

### Step 3: Train the CNN Model

```python
# Train the model
python lucid_cnn.py
```

## Configuration

### Counter-Based Sampling Parameters

- `keep_ratio`: Percentage of packets to keep (0.05 to 0.95)
- `min_per_key`: Minimum packets per traffic pattern (default: 100)

### CNN Model Parameters

- `PATIENCE`: Early stopping patience (default: 10)
- `DEFAULT_EPOCHS`: Maximum training epochs (default: 50)
- `hyperparamters`: Dictionary containing hyperparameter ranges for tuning

## Performance Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Balanced measure of precision and recall
- **TPR/FPR**: True/False Positive Rates
- **TNR/FNR**: True/False Negative Rates
- **FLOPs**: Computational complexity measurement
- **Inference Time**: Average prediction time per sample

## Dataset Support

The system supports various DDoS datasets:

- CIC-IDS2017
- CIC-IDS2018
- SYN2020
- DOS2019
- Custom datasets
