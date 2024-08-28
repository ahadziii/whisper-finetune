import argparse
import logging
import os
from datetime import datetime
import pandas as pd
import librosa
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer, process_words, Compose, ToLowerCase, RemoveWhiteSpace, RemoveMultipleSpaces, ReduceToListOfListOfWords

def setup_logging(log_directory):
    """
    Set up logging to both a file and the console.

    Parameters:
    - log_directory (str): Directory where the log file will be saved.
    
    Returns:
    - None
    """
    os.makedirs(log_directory, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_filename = f"benchmark_log_{current_datetime}.txt"
    log_file_path = os.path.join(log_directory, log_filename)

    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
    - argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Transcribe audio files and calculate WER.")
    parser.add_argument("--source_audio_path", type=str, required=True, help="Path to the file containing paths to WAV files")
    parser.add_argument("--source_transcription_path", type=str, required=True, help="Path to the file containing text transcriptions")
    parser.add_argument("--language", type=str, required=True, help="Language of the audio files")
    parser.add_argument("--log_directory", type=str, default='logs', help="Directory to save log files")
    parser.add_argument("--model_id", type=str, default="/mnt/disks/disk-1/whisper-large", help="Model ID or path")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for audio processing")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on")
    return parser.parse_args()

def load_model_and_processor(model_id, language, device):
    """
    Load the Whisper model and processor.

    Parameters:
    - model_id (str): Identifier or path to the pre-trained Whisper model.
    - language (str): Language of the audio files.
    - device (str): Device to run the model on (e.g., 'cuda:0').

    Returns:
    - processor (WhisperProcessor): Loaded processor for the Whisper model.
    - model (WhisperForConditionalGeneration): Loaded Whisper model.
    """
    logging.info(f"Loading model and processor: {model_id}")
    processor = WhisperProcessor.from_pretrained(model_id, language=language)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    return processor, model

def prepare_data(source_audio_path, source_transcription_path):
    """
    Prepare the dataset by reading audio and transcription file paths.

    Parameters:
    - source_audio_path (str): Path to the file containing paths to WAV files.
    - source_transcription_path (str): Path to the file containing text transcriptions.

    Returns:
    - Dataset: Dataset containing audio file paths and transcriptions.
    """
    logging.info(f"Reading audio and transcription file paths from: {source_audio_path} and {source_transcription_path}")
    wav_l = [temp.strip() for temp in open(source_audio_path).readlines()]
    text_l = [temp.strip() for temp in open(source_transcription_path).readlines()]
    df = pd.DataFrame(data={"file": wav_l, "audio": wav_l, "text": text_l})
    df.to_csv("test.csv", sep=",", index=False)
    logging.info("Data saved to test.csv")
    data_files = {'test': "test.csv"}
    dataset = load_dataset('csv', data_files=data_files)
    return dataset["test"]

def transcribe_and_calculate_wer(eval_data, processor, model, sample_rate, device, language):
    """
    Transcribe audio files and calculate the Word Error Rate (WER).

    Parameters:
    - eval_data (Dataset): Dataset containing audio file paths and transcriptions.
    - processor (WhisperProcessor): Processor for the Whisper model.
    - model (WhisperForConditionalGeneration): Whisper model for transcription.
    - sample_rate (int): Sample rate for audio processing.
    - device (str): Device to run the model on (e.g., 'cuda:0').
    - language (str): Language of the audio files.

    Returns:
    - pd.DataFrame: DataFrame containing the reference texts, transcriptions, and audio file paths.
    """
    ref, res, audi = [], [], []
    punc = '''!()-[]{};.:'"\,<>./?@#$%^&*_~'''
    k = 0

    logging.info("Starting transcription and WER calculation")
    for i in range(len(eval_data)):
        audio_sample = eval_data[i]
        if os.path.exists(audio_sample["file"]):
            audio, _ = librosa.load(audio_sample["file"], sr=sample_rate)
            test_str = ''.join([ele for ele in audio_sample["text"].lower() if ele not in punc])
            
            input_features = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features.to(device)
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True, language=language)
            transcribed_str = ''.join([ele for ele in transcription[0].lower() if ele not in punc])

            if transcribed_str and test_str:
                res.append(transcribed_str)
                ref.append(test_str)
                audi.append(audio_sample["file"])
            else:
                k += 1

            if i % 100 == 10:
                logging.info(f"Intermediate transcription result for file {audio_sample['file']}:\n{ref[i - k]} -> {res[i - k]}")

    transformation = Compose([
        ToLowerCase(),
        RemoveWhiteSpace(replace_by_space=True),
        RemoveMultipleSpaces(),
        ReduceToListOfListOfWords(word_delimiter=" ")
    ])

    out = process_words(ref, res)
    wer_result = wer(ref, res, truth_transform=transformation, hypothesis_transform=transformation) * 100

    logging.info(f"WER Whisper: {wer_result} Substitutions: {out.substitutions} Deletions: {out.deletions} Insertions: {out.insertions}")

    return pd.DataFrame({"Reference": ref, "Result": res, "Audio_File": audi})

def save_results(result_df, language, log_directory):
    """
    Save the transcription results to an Excel file.

    Parameters:
    - result_df (pd.DataFrame): DataFrame containing the transcription results.
    - language (str): Language of the audio files, used for naming the output file.
    - log_directory (str): Directory where the results file will be saved.

    Returns:
    - None
    """
    excel_filename = f"transcription_results_{language}.xlsx"
    excel_path = os.path.join(log_directory, excel_filename)

    try:
        result_df.to_excel(excel_path, index=False)
        logging.info(f"Transcription results saved to {excel_path}")
    except Exception as e:
        logging.error(f"Error saving transcription results to {excel_path}: {e}")

def main():
    """
    Main function to execute the transcription and WER calculation process.

    Parses command line arguments, sets up logging, loads the model and processor,
    prepares the data, performs transcription and WER calculation, and saves the results.
    
    Returns:
    - None
    """
    args = parse_arguments()
    setup_logging(args.log_directory)
    
    processor, model = load_model_and_processor(args.model_id, args.language, args.device)
    eval_data = prepare_data(args.source_audio_path, args.source_transcription_path)
    
    result_df = transcribe_and_calculate_wer(eval_data, processor, model, args.sample_rate, args.device, args.language)
    save_results(result_df, args.language, args.log_directory)

if __name__ == "__main__":
    main()
