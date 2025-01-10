# Automated Car Damage Classification System

## Project Overview
This project aims to develop an automated system for classifying different types of vehicle damage using computer vision and deep learning techniques. The system will be capable of identifying six distinct categories of car damage: cracks, scratches, flat tires, dents, glass shattering, and broken lamps. This technology has practical applications in the automotive insurance industry, where it can streamline damage assessment processes, reduce manual inspection time, and provide more consistent damage classifications.

## Technical Framework and Implementation

### Primary Framework: PyTorch Lightning
We will implement this project using PyTorch Lightning as our main framework. PyTorch Lightning was chosen for several key reasons:
- It provides a high-level interface that reduces boilerplate code while maintaining PyTorch's flexibility
- Built-in training optimization features like automatic batch size scaling and distributed training
- Integrated validation and testing loops
- Seamless integration with logging tools like Weights & Biases (wandb)

The framework will be incorporated through:
- Custom Lightning Modules for model definition and training logic
- Lightning DataModules for standardized data loading and processing
- Built-in training features for optimization and debugging

### Additional Framework: Albumentations Data Management and Augmentation

The initial dataset consists of car damage images organized into six categories:
1. Crack
2. Scratch
3. Flat tire
4. Dent
5. Glass shatter
6. Lamp broken

We will utilize the Albumentations library for robust image augmentation:
- Geometric transformations (rotation, flipping, scaling)
- Color augmentations (brightness, contrast, hue)
- Quality transforms (blur, noise, compression artifacts)

### Model Architecture

The primary model architecture will be based on ResNet, specifically:
- Initial implementation using ResNet-50 with weights pre-trained on ImageNet
- Modified final layers to accommodate our 6-class classification task
- Potential experimentation with other ResNet variants (ResNet-101, ResNet-152) based on performance
- All models will utilize ImageNet pre-training to leverage transfer learning benefits

Additional considerations:
- Transfer learning approach to leverage pre-trained weights
- Fine-tuning strategies for domain adaptation
- Potential ensemble methods combining multiple model variants

### Monitoring and Evaluation

We will implement a comprehensive evaluation system using TorchMetrics and Weights & Biases (wandb):

TorchMetrics Integration:
- Modular metric computation using TorchMetrics for reliable and efficient evaluation
- Implementation of specific metrics:
  - Accuracy (per-class and overall)
  - Precision (macro and weighted)
  - Recall (macro and weighted)
  - F1 Score (macro and weighted)
  - Area Under ROC Curve (AUC-ROC)
  - Confusion Matrix

Weights & Biases (wandb) Integration:
- Real-time training metrics monitoring
- Model performance comparisons
- Hyperparameter optimization
- Training artifact management
- Visualization of TorchMetrics results
- Custom metric logging and dashboards

Performance Tracking:
- Per-epoch metric computation and logging
- Real-time performance monitoring
- Cross-validation metrics
- Learning rate scheduling based on metric performance
- Early stopping using monitored metrics
- Best model checkpointing based on key metrics

This project combines modern deep learning frameworks with practical industry applications, aiming to create a robust and efficient system for automotive damage assessment.

## Setup Environment

1. Create a new conda environment using the provided `environment.yaml`:
```bash
conda env create -f environment.yaml
```

2. Activate the environment:
```bash
conda activate ml_ops_project
```

3. Verify the installation:
```bash
python --version  # Should show Python 3.11
```

4. Update environment (if needed after changing environment.yaml):
```bash
conda env update -f environment.yaml --prune
```

## Project structure
The directory structure of the project looks like this:
```txt
├── .github/ # Github actions and dependabot
│ ├── dependabot.yaml
│ └── workflows/
│ └── tests.yaml
├── configs/ # Configuration files
├── data/ # Data directory
│ ├── processed
│ └── raw
├── dockerfiles/ # Dockerfiles
│ ├── api.Dockerfile
│ └── train.Dockerfile
├── docs/ # Documentation
│ ├── mkdocs.yml
│ └── source/
│ └── index.md
├── models/ # Trained models
├── notebooks/ # Jupyter notebooks
├── reports/ # Reports
│ └── figures/
├── src/ # Source code
│ ├── project_name/
│ │ ├── __init__.py
│ │ ├── api.py
│ │ ├── data.py
│ │ ├── evaluate.py
│ │ ├── models.py
│ │ ├── train.py
│ │ └── visualize.py
└── tests/ # Tests
│ ├── __init__.py
│ ├── test_api.py
│ ├── test_data.py
│ └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── environment.yaml # Conda environment file
├── LICENSE
├── pyproject.toml # Python project file
├── README.md # Project README
├── requirements.txt # Project requirements
├── requirements_dev.txt # Development requirements
└── tasks.py # Project tasks
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
