# Whisper Fine-Tuning

This repository provides scripts and instructions for fine-tuning the Whisper automatic speech recognition (ASR) models on Spoken Oy data located on Google Cloud Platform. Whisper is an advanced ASR model developed by OpenAI, designed for robust transcription across a variety of languages and domains.


## Table of Contents
1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Usage](#usage)
    - [Dataset Preparation](#dataset-preparation)
    - [Benchmarking](#benchmarking)
    - [Fine-Tuning](#fine-tuning)
    - [Evaluate the Fine-Tuned Model](#evaluate-the-fine-tuned-model)
5. [Additional Resources](#additional-resources)


## Features

- **Dataset Download**: Download audio and transcription files from Google Cloud Storage.
- **Dataset Preparation**: Process and chunk audio files with their corresponding transcriptions & prepare data for Whisper fine-tuning
- **Fine-Tuning**: Fine-tune the Whisper model.
- **Evaluation**: Benchmark the performance on existing model & evaluate the performance of the fine-tuned model on a test set using Word Error Rate (WER).

## Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU
- [PyTorch](https://pytorch.org/get-started/locally/) (with CUDA support if using a GPU)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ahadziii/whisper-finetune.git
   cd whisper-finetune

2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt

4. **Set up Google Cloud credentials for accessing Google Cloud Storage**

   - Ask your admin to grant the necessary permissions to your VM’s service account.
   - You can find your VM’s service account name by running the following command on your Google Cloud CLI:

   ```bash
    # Replace INSTANCE_NAME with the name of your VM instance

    gcloud compute instances describe INSTANCE_NAME
   ```


## Usage

### Dataset Preparation

1. **Create an excel file with the following columns:**

   - `Audio/Video`: Name of the audio file in Google Cloud Storage.
   - `Transcript`: Name of the transcription file in Google Cloud Storage.
   - `Language`: Language of the audio & transcription file. eg `en` for English, `fi` for Finnish, `sv` for Swedish, `no` for Norwegian, `da` for Danish, `de` for German etc.

2. **Download the dataset from Google Cloud Storage:**

   - Run the `process_data/download_dataset.py` script to download media and their corresponding transcription files listed in the excel sheet from the Google Cloud Storage bucket. This script further processes the video files by converting them to audio.

   ```bash

    # bucket_name: name of the Google Cloud Storage bucket eg. examplebucketname.appspot.com
    # excel_file_path: path to the excel file containing the audio and transcript files
    # media_directory: path to the directory where the media files will be saved
    # transcription_directory: path to the directory where the transcription files will be saved
    # log_directory: path to the directory where the logs will be saved
    # excluded_emails: list of email addresses to exclude from the dataset when downloading (optional)
    # excluded_domains: list of domain names to exclude from the dataset when downloading (optional)

    python process_data/download_dataset.py \
        --bucket_name <bucket_name> \
        --excel_file_path <path_to_excel_file> \
        --media_directory <path_to_save_media> \
        --transcription_directory <path_to_save_transcription> \
        --log_directory <path_to_save_logs> \
        --excluded_emails <test@example.com> <another@example.com> \
        --excluded_domains <example.com> <anotherexample.com>
   ```

3. **Chunk the audio files:**

   - Run the `process_data/chunk_audio.py` script to process the audio and transcription files (in SRT and VTT subtitle formats) listed in the Excel sheet. This script will convert the audio to a single channel, split it into chunks, and save the corresponding transcript segments. The processed audio and transcripts will also be organized by their respective languages specified in the excel file.

   ```bash

    # excel_path: path to the excel file containing the audio and transcript files
    # base_audio_path: path to the directory containing the audio
    # base_transcript_path: path to the directory containing the transcript files
    # output_dir: path to the directory where the audio and transcription chunks will be saved
    # log_directory: path to the directory where the logs will be saved
   
    python process_data/chunk_data.py \
        --excel_path <path_to_excel_file> \
        --base_audio_path <path_to_audio_files> \
        --base_transcript_path <path_to_transcript_files> \
        --output_dir <path_to_output> \
        --log_directory <path_to_log_directory>
   ```

4. **Processing Media and Transcriptions:**
    - Once the audio files have been chunked, make sure to split the data into training and evaluation sets. This can be done by moving the audio and transcription chunks into separate directories for training and evaluation.

    - Run the `process_data/process_media_transcription.py` script individually for both training directory and evaluation directory to process the chunked media and transcription files. This script will match the media files with their corresponding transcription files, then save the media file paths and the transcription content to a specified output directory.
    
    ```bash

    # media_directory: path to the directory containing the audio chunks
    # transcription_directory: path to the directory containing the transcript chunks
    # output: path to the output directory where the processed data will be saved
    # language: language of the data
    
    python process_data/process_media_transcriptions.py 
        --media_directory </path/to/lang/media_dir> \
        --transcription_directory </path/to/lang/transcription_dir> \
        --output </path/to/output_dir> \
        --language en
    ```

    NOTE: Since our dataset isn't available over huggingface. To fine-tune whisper models or evaluate them on such datasets, a preliminary data preparation is needed to make them compatible with the huggingface's sequence-to-sequence training pipeline. The above script converts the data into two files, one containing the paths to the audio files and the other containing the transcriptions as shown below:

    eg. `media_lang.txt`:
    
    ```bash
    <absolute path to the audio file-1>
    <absolute path to the audio file-2>
    ...
    <absolute path to the audio file-N>
    ```

    eg. `transcription_lang.txt`:

    ```bash
    <Transcription (ground truth) corresponding to the audio file-1>
    <Transcription (ground truth) corresponding to the audio file-2>
    ...
    <Transcription (ground truth) corresponding to the audio file-N>
    ```

    Once the data has been organized in this manner, the script named `process_data/prepare_whisper_data.py` is run on both train and evaluation data to convert the data into the format expected by sequence-to-sequence pipeline of huggingface.

    Following is a sample command to convert the data into the desired format:
    
    ```bash

    # source_audio_path: path to the file containing the paths to the audio
    # source_transcription_path: path to the file containing the transcriptions
    # output_data_dir: path to the output directory where the processed data will be saved
    # language: language of the data
 
    python process_data/prepare_whisper_data.py
        --source_audio_path <path/to/audio_paths.txt> \
        --source_transcription_path <path/to/transcriptions.txt> \
        --output_data_dir <path/to/output/directory> \
        --language en
    ```


### Benchmarking

1. **Benchmark the performance of the existing Whisper model:**

   - Run the `evaluate/evaluate_model.py` script on the evaluation data to benchmark the performance of the existing Whisper model on a test set using Word Error Rate (WER).

   ```bash
    # wav_path: path to the file containing the paths to the audio files
    # text_path: path to the file containing the transcriptions
    # log_directory: path to the directory where the logs will be saved
    # model_id: model name ie. this can be a model name on Hugging Face or a path to the model on disk
    # language: language of the data


   python evaluate/evaluate_model.py \
    --wav_path path/to/wav.txt \
    --text_path path/to/text.txt \
    --language en \
    --log_directory path/to/log/directory \
    --model_id /path/to/model \
    --sample_rate 16000 \
    --device cuda:0
   ```

### Fine-Tuning

1. **Fine-tune the Whisper model:**

   - Run the `train/fine-tune_on_custom_dataset.py` script to fine-tune the Whisper model on a custom dataset.

   ```bash

    # model_name: model name to be finetuned ie. this can be a model name on Hugging Face or a path to the model on disk
    # sampling_rate: sampling rate of the audio files
    # num_proc: number of processes to use for data loading
    # train_strategy: training strategy to use (epoch or steps)
    # learning_rate: learning rate for the optimizer
    # warmup: number of warmup steps
    # train_batchsize: batch size for training
    # eval_batchsize: batch size for evaluation
    # num_epochs: number of epochs to train the model
    # resume_from_ckpt: path to the checkpoint to resume training from
    # output_dir: path to the output directory where the model will be saved
    # train_datasets: list of paths to the training datasets
    # eval_datasets: list of paths to the evaluation datasets


    ngpu=1  # number of GPUs to perform distributed training on.

    torchrun --nproc_per_node=${ngpu} train/fine-tune_on_custom_dataset.py \
        --model_name openai/whisper-large-v3 \
        --sampling_rate 16000 \
        --num_proc 1 \
        --train_strategy epoch \
        --learning_rate 1e-6 \
        --warmup 1000 \
        --train_batchsize 2 \
        --eval_batchsize 2 \
        --num_epochs 2 \
        --resume_from_ckpt None \
        --output_dir op_dir_epoch \
        --train_datasets output_data_directory/train_dataset_1 output_data_directory/train_dataset_2 \
        --eval_datasets output_data_directory/eval_dataset_1 output_data_directory/eval_dataset_2 output_data_directory/eval_dataset_3
   ```

   The datasets being passed as parameters through the train_datasets and eval_datasets arguments should have been from the output directories generated through the data preparation stage. Multiple datasets can be used as a part of the fine-tuning process. These datasets would be concatenated and shuffled at the time of dataset preparation.

   While all of the arguments are set with default options, one is encouraged to look into the file to customize the training hyperparameters in such a way that it suits the amount of data at hand and the size of the model being used.



## Evaluate the Fine-Tuned Model

1. **Evaluate the fine-tuned model:**

   - Run the `evaluate/evaluate_model.py` script to evaluate the performance of the fine-tuned model on a test set using Word Error Rate (WER).

   ```bash
   python evaluate/evaluate_model.py \
    --wav_path path/to/wav.txt \
    --text_path path/to/text.txt \
    --language en \
    --log_directory logs \
    --model_id /path/to/model \
    --sample_rate 16000 \
    --device cuda:0
   ```
   This script will calculate the Word Error Rate (WER) on the test set to assess the model’s performance.


## Additional Resources

- [Whisper: A Speech Recognition System for Everyone](https://arxiv.org/abs/2110.13979)
- [OpenAI Blog Post](https://www.openai.com/blog/whisper/)
- [OpenAI GitHub Repository](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/transformers/)



