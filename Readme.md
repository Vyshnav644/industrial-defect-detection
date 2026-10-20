# Industrial Steel Surface Defect Detection

## Overview

This project uses deep learning and computer vision to classify
steel surface images into six defect categories.

Automated defect classification can support industrial quality
inspection by identifying potential surface defects and helping
reduce reliance on manual inspection.

The project implements a baseline Convolutional Neural Network
(CNN) and a ResNet18 transfer learning model using PyTorch.
A Streamlit application provides an interface for uploading
steel surface images and viewing predictions.

## Objectives

- Explore and understand an industrial surface defect dataset.
- Build a baseline CNN for image classification.
- Apply transfer learning using ResNet18.
- Evaluate model performance using classification metrics.
- Save the trained model for reuse.
- Develop a reusable inference module.
- Build a Streamlit application for image-based predictions.

## Dataset

The project uses the NEU-DET steel surface defect dataset.

The dataset contains six defect categories:

| Class | Description |
|---|---|
| Crazing | Fine cracks on the steel surface |
| Inclusion | Foreign material inclusions |
| Patches | Irregular surface patches |
| Pitted Surface | Small pits or cavities |
| Rolled-in Scale | Oxide scale embedded during rolling |
| Scratches | Linear surface marks |

### Dataset Split

| Split | Number of Images |
|---|---:|
| Training | 1,440 |
| Validation | 360 |
| Total | 1,800 |

Each class contains 240 training images and 60 validation images.

The dataset is organized into training and validation directories.
Images are loaded using torchvision's ImageFolder utility.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow
- Scikit-learn
- Streamlit

## Models

### 1. Baseline CNN

A custom CNN was implemented as an initial image classification
model.

The architecture includes convolutional layers, ReLU activations,
max pooling, and fully connected classification layers.

Validation accuracy: **71.94%**

### 2. ResNet18 Transfer Learning

ResNet18 was used as the second model.

The original classification layer was replaced with a new layer
for the six steel defect categories.

The model was trained and evaluated using PyTorch.

Validation accuracy: **98.61%**

### Model Comparison

| Model | Validation Accuracy |
|---|---:|
| Baseline CNN | 71.94% |
| ResNet18 | 98.61% |

ResNet18 achieved substantially higher validation accuracy than
the baseline CNN in the current experiment.

These results describe performance on the validation split.
They should not be interpreted as a guarantee of performance
on images from different industrial environments.

## Evaluation

Model performance was examined using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The ResNet18 classification report showed strong performance
across all six classes on the validation dataset.

The evaluation also highlighted some misclassifications,
including confusion between inclusion and pitted surface,
and between scratches and rolled-in scale.

### Data Quality Checks

Image hashing was used to investigate duplicate image contents.

The investigation identified one duplicate pair within the
training dataset. No identical image contents were found across
the training and validation splits in the performed comparison.

The duplicate pair should be reviewed and handled appropriately
when finalizing the dataset and reported results.

## Project Structure

```text
Industrial_Defect_Detection/
├── App/
│   └── app.py
├── config/
│   └── config.py
├── Data/
│   ├── External_test/
│   ├── Processed/
│   ├── Raw/
│   └── Splits/
├── Evaluation/
├── Models/
│   ├── class_mapping.json
│   ├── model.py
│   ├── predict.py
│   └── resnet18_neu_det.pth
├── Notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_cnn_training.ipynb
├── Preprocessing/
│   └── preprocessing.py
├── Results/
├── Training/
│   └── train.py
├── Visualization/
├── Readme.md
└── Requirements.txt
```

Some directories are reserved for organizing additional project
artifacts and outputs.

## Running the Application

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Industrial_Defect_Detection
```

Replace the placeholder with the actual repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r Requirements.txt
```

### 4. Launch the Application

From the project root:

```bash
python -m streamlit run App/app.py
```

Open the local address printed by Streamlit in your browser.

### 5. Make a Prediction

1. Upload a JPG, JPEG, or PNG steel surface image.
2. Click **Predict Defect**.
3. Review the predicted defect category and confidence score.

The application loads the saved model rather than retraining it.

## Saved Model and Inference

The trained model parameters are stored in:

`Models/resnet18_neu_det.pth`

The class mapping is stored in:

`Models/class_mapping.json`

The reusable prediction function is implemented in:

`Models/predict.py`

The application uses this function to preprocess uploaded images
and generate predictions.

## Limitations

- The current evaluation uses a limited dataset.
- Validation accuracy does not establish real-world performance.
- Confidence scores are not guarantees of prediction correctness.
- Independent testing on appropriately labelled industrial images
  is required to evaluate generalization.
- The duplicate training image pair should be investigated.
- The application is a demonstration and is not a certified
  industrial inspection system.

## Future Improvements

- Conduct additional dataset quality checks.
- Investigate and resolve duplicate training images.
- Evaluate the model on independently sourced steel images.
- Compare alternative architectures and preprocessing strategies.
- Improve reproducibility through configuration and experiment
  tracking.
- Add prediction history and confidence visualizations.
- Package and deploy the application for broader access.

## Disclaimer

This project is intended for educational and portfolio purposes.
Predictions should not be used as the sole basis for industrial
quality-control decisions without appropriate independent
validation.

## Author

Developed as a computer vision and deep learning portfolio project.