# Meena - Enterprise AI Pipeline

This repository contains an enterprise-grade CI/CD pipeline for training, benchmarking, and deploying the Meena conversational AI model.

## 🚀 Features

- **Automated CI/CD**: The entire process from training to deployment is automated using GitHub Actions.
- **Advanced Training**: Utilizes LoRA for efficient fine-tuning of models like DialoGPT.
- **Bengali & English Support**: The training data and scripts are designed for multilingual conversations.
- **Benchmarking**: Includes a dedicated job to benchmark model performance.
- **Auto-generated Model Cards**: The `generate_model_card.py` script creates detailed README files for the published models.
- **Smart Change Detection**: The workflow only runs jobs relevant to the files that have changed.

##  workflow

The core logic is defined in `.github/workflows/auto-train-publish.yml`. This workflow is triggered on pushes to `main` or `develop`, on pull requests to `main`, or can be dispatched manually.

It consists of the following jobs:
1.  `detect-changes`: Determines which parts of the pipeline need to run.
2.  `train`: Trains the model using the `train.py` script.
3.  `benchmark`: Benchmarks the trained model using `benchmark.py`.
4.  `publish`: Publishes the model to the Hugging Face Hub and creates a GitHub Release.
5.  `test`: Runs a smoke test against the deployed model on the Hugging Face Inference API.
6.  `notify`: Sends a notification about the pipeline status.

## Usage

Pushing changes to the relevant files will automatically trigger the pipeline. You can also trigger it manually from the Actions tab in the GitHub repository.
