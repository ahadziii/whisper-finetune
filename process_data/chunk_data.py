import os
import logging
import pandas as pd
from pydub import AudioSegment
import pysrt
import webvtt
from datetime import datetime
import argparse


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
    log_filename = f"chunk_data_log_{current_datetime}.txt"
    log_file_path = os.path.join(log_directory, log_filename)

    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


def check_file_exists(file_path):
    """
    Check if a file exists at the given path.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - bool: True if file exists, False otherwise.
    """
    logging.info(f"Checking if file exists: {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"File does not exist: {file_path}")
        return False
    logging.info(f"File exists: {file_path}")
    return True


def convert_to_one_channel(audio_path):
    """
    Convert audio to one channel (mono).

    Parameters:
    - audio_path (str): Path to the audio file.

    Returns:
    - AudioSegment: Mono audio segment.
    """
    logging.info(f"Converting audio to one channel: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)
    return audio


def parse_srt(file_path):
    """
    Parse an SRT file and return a list of chunks with start time, end time, and text.

    Parameters:
    - file_path (str): Path to the SRT file.

    Returns:
    - list: List of tuples containing start time, end time, and text.
    """
    logging.info(f"Parsing SRT file: {file_path}")
    subs = pysrt.open(file_path)
    chunks = [(sub.start.ordinal, sub.end.ordinal, sub.text) for sub in subs]
    return chunks


def parse_vtt(file_path):
    """
    Parse a VTT file and return a list of chunks with start time, end time, and text.

    Parameters:
    - file_path (str): Path to the VTT file.

    Returns:
    - list: List of tuples containing start time, end time, and text.
    """
    logging.info(f"Parsing VTT file: {file_path}")
    subs = webvtt.read(file_path)
    chunks = [(cue.start_in_seconds * 1000, cue.end_in_seconds * 1000, cue.text) for cue in subs]
    return chunks


def save_chunks(audio, chunks, media_directory, transcription_directory, output_dir, language_code):
    """
    Save audio chunks and transcripts to the output directories, grouped by language.

    Parameters:
    - audio (AudioSegment): The full audio segment.
    - chunks (list): List of tuples with start time, end time, and text.
    - media_directory (str): Path to the base audio file.
    - transcription_directory (str): Path to the base transcript file.
    - output_dir (str): Base output directory to save chunks.
    - language_code (str): Language code to group the chunks by.

    Returns:
    - None
    """
    logging.info(f"Saving chunks for audio: {media_directory} and transcript: {transcription_directory}")
    base_audio_name = os.path.splitext(os.path.basename(media_directory))[0]

    # Create language-specific directories under the output directory
    language_media_dir = os.path.join(output_dir, language_code, 'media')
    language_transcript_dir = os.path.join(output_dir, language_code, 'transcriptions')
    os.makedirs(language_media_dir, exist_ok=True)
    os.makedirs(language_transcript_dir, exist_ok=True)

    for i, (start, end, text) in enumerate(chunks):
        chunk_audio_path = os.path.join(language_media_dir, f"{base_audio_name}_{i+1}.wav")
        chunk_transcript_path = os.path.join(language_transcript_dir, f"{base_audio_name}_{i+1}.txt")

        if os.path.exists(chunk_audio_path) and os.path.exists(chunk_transcript_path):
            logging.info(f"Chunk {i+1} already exists. Skipping. Audio: {chunk_audio_path}, Transcript: {chunk_transcript_path}")
            continue

        chunk_audio = audio[start:end]
        chunk_audio.export(chunk_audio_path, format="wav")
        with open(chunk_transcript_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logging.info(f"Saved chunk {i+1} in {language_code}: {chunk_audio_path}, {chunk_transcript_path}")


def process_files(excel_path, media_directory, transcription_directory, output_dir):
    """
    Process files listed in the Excel sheet, chunk them, and save them grouped by language.

    Parameters:
    - excel_path (str): Path to the Excel file containing the list of audio and transcript files.
    - media_directory (str): Base directory containing audio files.
    - transcription_directory (str): Base directory containing transcript files.
    - output_dir (str): Base directory to save processed chunks, organized by language.

    Returns:
    - None
    """
    df = pd.read_excel(excel_path)

    for index, row in df.iterrows():
        audio_file_name = os.path.splitext(row['Audio/Video'])[0] + '.wav'
        audio_path = os.path.join(media_directory, audio_file_name)
        transcript_path = os.path.join(transcription_directory, row['Transcript'])
        language_code = row['Language']  

        if not check_file_exists(audio_path) or not check_file_exists(transcript_path):
            logging.error(f"Files not found: {audio_path}, {transcript_path}")
            continue

        try:
            audio = convert_to_one_channel(audio_path)
            if transcript_path.endswith('.srt'):
                chunks = parse_srt(transcript_path)
            elif transcript_path.endswith('.vtt'):
                chunks = parse_vtt(transcript_path)
            else:
                logging.error(f"Unsupported transcript format: {transcript_path}")
                continue
            
            save_chunks(audio, chunks, audio_path, transcript_path, output_dir, language_code)

        except Exception as e:
            logging.error(f"Error processing files {audio_path} and {transcript_path}: {str(e)}")


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
    - argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Process audio and transcript files to create chunks.")
    parser.add_argument("--excel_path", type=str, required=True, help="Path to the Excel file listing audio and transcript files")
    parser.add_argument("--media_directory", type=str, required=True, help="Base directory containing audio files")
    parser.add_argument("--transcription_directory", type=str, required=True, help="Base directory containing transcript files")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save processed chunks, grouped by language")
    parser.add_argument("--log_directory", type=str, default='logs', help="Directory to save log files")
    return parser.parse_args()


def main():
    """
    Main function to execute the file processing.

    Parses command line arguments, sets up logging, and processes files to create audio and transcript chunks,
    organized by language.

    Returns:
    - None
    """
    try:
        args = parse_arguments()
        setup_logging(args.log_directory)

        logging.info("Starting the file processing")
        process_files(args.excel_path, args.media_directory, args.transcription_directory, args.output_dir)
        logging.info("File processing completed")

    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")


if __name__ == '__main__':
    main()