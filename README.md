# ml_ops_project
A short description of the project.

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