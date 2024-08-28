# Whisper Fine-Tuning

This repository provides scripts and instructions for fine-tuning the Whisper automatic speech recognition (ASR) models on Spoken Oy data located on Google Cloud Platform. Whisper is an advanced ASR model developed by OpenAI, designed for robust transcription across a variety of languages and domains.


## Table of Contents
1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Setting Up a VM on Google Cloud Platform (GCP) with GPU Support](#setting-up-a-vm-on-google-cloud-platform-gcp-with-gpu-support)
    - [Create the Instance](#create-the-instance)
    - [Install Google Cloud SDK](#install-google-cloud-sdk)
    - [SSH into the VM Instance](#ssh-into-the-vm-instance)
    - [Install Miniconda on the VM](#install-miniconda-on-the-vm)
    - [Create and Activate a New Conda Environment](#create-and-activate-a-new-conda-environment)
    - [Format and Mount the attached New Disk](#format-and-mount-the-attached-new-disk)
4. [Installation](#installation)
5. [Usage](#usage)
    - [Dataset Preparation](#dataset-preparation)
    - [Benchmarking](#benchmarking)
    - [Fine-Tuning](#fine-tuning)
    - [Evaluate the Fine-Tuned Model](#evaluate-the-fine-tuned-model)
6. [Additional Resources](#additional-resources)
    - [Running Commands in the Background](#running-commands-in-the-background)
    - [Running Multiple Commands](#running-multiple-commands)
    - [References](#references)



## Features

- **Dataset Download**: Download audio and transcription files from Google Cloud Storage.
- **Dataset Preparation**: Process and chunk audio files with their corresponding transcriptions & prepare data for Whisper fine-tuning
- **Fine-Tuning**: Fine-tune the Whisper model.
- **Evaluation**: Benchmark the performance on existing model & evaluate the performance of the fine-tuned model on a test set using Word Error Rate (WER).

## Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU
- [PyTorch](https://pytorch.org/get-started/locally/) (with CUDA support if using a GPU)

## Setting Up a VM on Google Cloud Platform (GCP) with GPU Support

This guide will walk you through the steps required to set up a Virtual Machine (VM) on Google Cloud Platform (GCP) with GPU support, install necessary tools, and configure the environment for your project.

Note: You can skip this step if you already have a VM set up on GCP with GPU support or if you are using a different cloud provider.

### Create the Instance

1. **Log in to Google Cloud Console**: Visit the [Google Cloud Console](https://console.cloud.google.com/).

2. **Create a New VM Instance**:
   - Navigate to the "Compute Engine" section and select "VM Instances."
   - Click on "Create Instance."
   - Choose your desired machine type and configure the GPU and memory:
     - Under the "Machine configuration" section, select a machine type that suits your needs.
     - In the "GPU" section, add the necessary GPU by selecting the appropriate type and number of GPUs.
     - Configure the memory and other settings as needed.
   - Under "Boot disk," switch to the "Custom images" tab and select a Deep Learning Image (e.g., `Deep Learning VM with CUDA 11.8 M124`).
   - Make sure to allow HTTP/HTTPS traffic under the "Firewall" section if required by your application.
   - Under "Advanced Options", go to disk and add a new disk with the size of your choice. 

3. **Create the Instance**: After configuring all settings, click "Create" to launch the instance.

### Install Google Cloud SDK

1. **Download and Install the gcloud CLI on your local machine**:
   - Visit the [Google Cloud SDK installation page](https://cloud.google.com/sdk/docs/install) and follow the instructions for your operating system.

2. **Initialize the gcloud CLI**:
   - After installation, open your terminal and run:
     ```bash
     gcloud init
     ```
   - Follow the prompts to authenticate and set the default project.

### SSH into the VM Instance

1. **Connect to the VM**:
   - Use the following command to SSH into your VM instance:
     ```bash
     gcloud compute ssh --zone "your-zone" "your-instance-name"
     ```
   - Replace `"your-zone"` with your VM's zone and `"your-instance-name"` with the name of your instance.

### Install Miniconda on the VM

1. **Download the Miniconda Installer**:
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   ```

2. **Run the Installer**:
   ```bash
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
   - Follow the prompts to complete the installation.

3. **Initialize Conda**:
   
   ```bash
   echo $SHELL
   ```
   - If the output is `/bin/bash`, run the following command:
     ```bash
     source ~/.bashrc
     ```
   - If the output is `/bin/zsh`, run the following command:
     ```bash
       source ~/.zshrc
       ```
   - Initialize Conda by running:
       ```bash
       eval "$(/home/your_username/miniconda3/bin/conda shell.bash hook)"
       conda activate base
       ```
   - Ensure that Conda is initialized upon login by adding the following to your `.bashrc` or `.zshrc`:
       ```bash
       echo 'eval "$(/home/your_username/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc
       ```
         or
       ```bash
       echo 'eval "$(/home/your_username/miniconda3/bin/conda shell.zsh hook)"' >> ~/.zshrc
       ```

### Create and Activate a New Conda Environment

1. **Create a New Environment**:
   ```bash
   conda create -n myenv python=3.8
   ```
2. **Activate the Environment**:
   ```bash
   conda activate myenv
   ```

### Format and Mount the attached New Disk

If you've attached a new disk to your VM, follow these steps to format and mount it:

1. **List All Disks to Identify the New Disk**:
   ```bash
   lsblk
   ```
2. **Format the Disk (if new and unformatted)**:
   ```bash
   sudo mkfs.ext4 -F /dev/name_of_disk
   ```
3. **Create a Mount Point**:
   ```bash
   sudo mkdir -p /path/to/mount/point
   ```
4. **Mount the Disk**:
   ```bash
   sudo mount /dev/name_of_disk /path/to/mount/point
   ```
5. **Verify the Mount**:
   ```bash
   df -h /path/to/mount/point
   ```
   You should see the disk mounted at the specified path.

6. **Automatically Mount the Disk on Boot**:
   - Add an entry to `/etc/fstab` to automatically mount the disk on boot:
     ```bash
     echo "/dev/name_of_disk /path/to/mount/point ext4 defaults 0 0" | sudo tee -a /etc/fstab
     ```
   - Replace `"name_of_disk"` and `"path/to/mount/point"` with the appropriate values.

7. **Change the Ownership of the Mounted Disk**:
   - Change the ownership of the mounted disk to your user:
     ```bash
     sudo chown -R $USER:$USER /path/to/mount/point
     ```

8. **Verify the Ownership**:
   - Verify that the ownership has been changed:
     ```bash
     ls -l /path/to/mount/point
     ```
   - You should see your username as the owner of the mounted disk.

9. **Navigate to the Mounted Disk**:
   - You can now navigate to the mounted disk and use it for storing data or running scripts.
   ```bash
   cd /path/to/mount/point
   ```


## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ahadziii/whisper-finetune.git
   cd whisper-finetune
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt

3. **Set up Google Cloud credentials for accessing Google Cloud Storage**

   - Ask your admin to grant the necessary permissions to your VM’s service account.
   - You can find your VM’s service account name by running the following command on your Google Cloud CLI:

   ```bash
    # Replace INSTANCE_NAME with the name of your VM instance

    gcloud compute instances describe INSTANCE_NAME
   ```

4. **Set up huggingface token for model download**

   - Create an account on huggingface and generate an API token.
   - Run the following command to set up the token:

   ```bash
    huggingface-cli login
   ```
   - Enter the API token when prompted.


## Usage

### Dataset Preparation



1. **Create an excel file with the following columns:**

   - `Audio/Video`: Name of the audio file in Google Cloud Storage.
   - `Transcript`: Name of the transcription file in Google Cloud Storage.
   - `Language`: Language of the audio & transcription file. eg `en` for English, `fi` for Finnish, `sv` for Swedish, `no` for Norwegian, `da` for Danish, `de` for German etc.

   There is an example excel file in `example_template.xlsx` in the `excel_template` directory. You can download this file and fill in the audio and transcription file names along with the language code.


2. **Download the dataset from Google Cloud Storage:**

   - Run the `process_data/download_data.py` script to download media and their corresponding transcription files listed in the excel sheet from the Google Cloud Storage bucket. This script further processes the video files by converting them to audio.

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
    # media_directory: path to the directory containing the audio
    # transcription_directory: path to the directory containing the transcript files
    # output_dir: path to the directory where the audio and transcription chunks will be saved
    # log_directory: path to the directory where the logs will be saved
   
    python process_data/chunk_data.py \
        --excel_path <path_to_excel_file> \
        --media_directory <path_to_audio_files> \
        --transcription_directory <path_to_transcript_files> \
        --output_dir <path_to_output> \
        --log_directory <path_to_log_directory>
   ```

4. **Processing Media and Transcriptions:**
    - After chunking the audio files, divide the data into training and evaluation sets. This involves moving a portion of the audio and their `corresponding` transcription chunks into distinct directories for training and evaluation purposes.

    - Run the `process_data/process_media_transcription.py` script separately for both the training and evaluation directories for each language (if the dataset contains multiple languages). This script will pair the media files with their corresponding transcription files and save the media file paths along with the transcription content to a designated output directory.
    
    ```bash

    # media_directory: path to the directory containing the audio chunks
    # transcription_directory: path to the directory containing the transcript chunks
    # output: path to the output directory where the processed data will be saved
    # language: language of the data
    
    python process_data/process_media_transcription.py \
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
 
    python process_data/prepare_whisper_data.py \
        --source_audio_path <path/to/audio_paths.txt> \
        --source_transcription_path <path/to/transcriptions.txt> \
        --output_data_dir <path/to/train/output/directory> \
        --language en
    ```


### Benchmarking

1. **Benchmark the performance of the existing Whisper model:**

   - Run the `evaluate/evaluate_model.py` script on the evaluation data to benchmark the performance of the existing Whisper model on a test set using Word Error Rate (WER).

   ```bash
    # source_audio_path: path to the file containing the paths to the audio files
    # source_transcription_path: path to the file containing the transcriptions
    # log_directory: path to the directory where the logs will be saved
    # model_id: model name ie. this can be a model name on Hugging Face or a path to the model on disk
    # language: language of the data


   python evaluate/evaluate_model.py \
    --source_audio_path path/to/media_lang.txt \
    --source_transcription_path path/to/transcription_lang.txt \
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
    # train_datasets: list of paths to the training datasets prepared from `prepare_whisper_data.py`
    # eval_datasets: list of paths to the evaluation datasets prepared from `prepare_whisper_data.py`


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
    --source_audio_path path/to/wav.txt \
    --source_transcription_path path/to/text.txt \
    --language en \
    --log_directory logs \
    --model_id /path/to/model \
    --sample_rate 16000 \
    --device cuda:0
   ```
   This script will calculate the Word Error Rate (WER) on the test set to assess the model’s performance.


## Additional Resources

### Running Commands in the Background

You can attach `> name_of_file.log 2>&1 & disown` to the end of a command to detach it from a session, run it in the background and send the output to the log file `name_of_file.log` specified. The `2>&1` redirects both standard output and standard error to the log file. The file will be created in the location where the command was run.

```bash
python script.py > name_of_file.log 2>&1 & disown
```

### Running Multiple Commands

You can also run a group of commands in the background by the `nohup` command. For example, say you want to `process media transcriptions` for three different languages. You can use the following command:

```bash
nohup bash -c "
python process_data/process_media_transcription.py \
        --media_directory </path/to/en/media_dir> \
        --transcription_directory </path/to/en/transcription_dir> \
        --output </path/to/output_dir> \
        --language en

python process_data/process_media_transcription.py \
        --media_directory </path/to/fi/media_dir> \
        --transcription_directory </path/to/fi/transcription_dir> \
        --output </path/to/output_dir> \
        --language fi

python process_data/process_media_transcription.py \
        --media_directory </path/to/sv/media_dir> \
        --transcription_directory </path/to/sv/transcription_dir> \
        --output </path/to/output_dir> \
        --language sv
" &
disown
```

This will run all the specified commands in the background as a detached session and output the results to the `nohup.out` file in the location where the command was run.


### References

- [Whisper: A Speech Recognition System for Everyone](https://arxiv.org/abs/2110.13979)
- [OpenAI Blog Post](https://www.openai.com/blog/whisper/)
- [OpenAI GitHub Repository](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/transformers/)



