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


def save_chunks(audio, chunks, base_audio_path, base_transcript_path, output_media_dir, output_transcript_dir):
    """
    Save audio chunks and transcripts to the output directories.

    Parameters:
    - audio (AudioSegment): The full audio segment.
    - chunks (list): List of tuples with start time, end time, and text.
    - base_audio_path (str): Path to the base audio file.
    - base_transcript_path (str): Path to the base transcript file.
    - output_media_dir (str): Directory to save audio chunks.
    - output_transcript_dir (str): Directory to save transcript chunks.

    Returns:
    - None
    """
    logging.info(f"Saving chunks for audio: {base_audio_path} and transcript: {base_transcript_path}")
    base_audio_name = os.path.splitext(os.path.basename(base_audio_path))[0]
    
    for i, (start, end, text) in enumerate(chunks):
        chunk_audio_path = os.path.join(output_media_dir, f"{base_audio_name}_{i+1}.wav")
        chunk_transcript_path = os.path.join(output_transcript_dir, f"{base_audio_name}_{i+1}.txt")

        # Skip saving if chunk files already exist
        if os.path.exists(chunk_audio_path) and os.path.exists(chunk_transcript_path):
            logging.info(f"Chunk {i+1} already exists. Skipping. Audio: {chunk_audio_path}, Transcript: {chunk_transcript_path}")
            continue

        chunk_audio = audio[start:end]
        chunk_audio.export(chunk_audio_path, format="wav")
        with open(chunk_transcript_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        logging.info(f"Saved chunk {i+1}: {chunk_audio_path}, {chunk_transcript_path}")


def process_files(excel_path, base_audio_path, base_transcript_path, output_media_dir, output_transcript_dir):
    """
    Process files listed in the Excel sheet.

    Parameters:
    - excel_path (str): Path to the Excel file containing the list of audio and transcript files.
    - base_audio_path (str): Base directory containing audio files.
    - base_transcript_path (str): Base directory containing transcript files.
    - output_media_dir (str): Directory to save processed audio chunks.
    - output_transcript_dir (str): Directory to save processed transcript chunks.

    Returns:
    - None
    """
    df = pd.read_excel(excel_path)

    for index, row in df.iterrows():
        audio_file_name = os.path.splitext(row['Audio/Video'])[0] + '.wav'
        audio_path = os.path.join(base_audio_path, audio_file_name)
        transcript_path = os.path.join(base_transcript_path, row['Transcript'])

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
            
            save_chunks(audio, chunks, audio_path, transcript_path, output_media_dir, output_transcript_dir)

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
    parser.add_argument("--base_audio_path", type=str, required=True, help="Base directory containing audio files")
    parser.add_argument("--base_transcript_path", type=str, required=True, help="Base directory containing transcript files")
    parser.add_argument("--output_media_dir", type=str, required=True, help="Directory to save processed audio chunks")
    parser.add_argument("--output_transcript_dir", type=str, required=True, help="Directory to save processed transcript chunks")
    parser.add_argument("--log_directory", type=str, default='logs', help="Directory to save log files")
    return parser.parse_args()


def main():
    """
    Main function to execute the file processing.

    Parses command line arguments, sets up logging, and processes files to create audio and transcript chunks.
    
    Returns:
    - None
    """
    try:
        args = parse_arguments()
        setup_logging(args.log_directory)

        logging.info("Starting the file processing")
        process_files(args.excel_path, args.base_audio_path, args.base_transcript_path, args.output_media_dir, args.output_transcript_dir)
        logging.info("File processing completed")

    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")


if __name__ == '__main__':
    main()



# python chunk_data.py --excel_path <path_to_excel_file> \
#                                     --base_audio_path <path_to_audio_files> \
#                                     --base_transcript_path <path_to_transcript_files> \
#                                     --output_media_dir <path_to_output_audio_chunks> \
#                                     --output_transcript_dir <path_to_output_transcript_chunks> \
#                                     --log_directory <path_to_log_directory>