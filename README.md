# Whisper Fine-Tuning

This repository provides scripts and instructions for fine-tuning the Whisper automatic speech recognition (ASR) model on custom datasets. Whisper is an advanced ASR model developed by OpenAI, designed for robust transcription across a variety of languages and domains.

## Features

- **Dataset Preparation**: Tools to preprocess your audio and transcription datasets.
- **Fine-Tuning**: Scripts to fine-tune the Whisper model using PyTorch.
- **Inference**: Run inference using your fine-tuned model.
- **Evaluation**: Evaluate the performance of the fine-tuned model on a test set.

## Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU (optional but recommended)
- [PyTorch](https://pytorch.org/get-started/locally/) (with CUDA support if using a GPU)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ahadziii/whisper-finetune.git
   cd whisper-finetune