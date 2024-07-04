import os
import logging
import pandas as pd
from pydub import AudioSegment
import pysrt
import webvtt
from datetime import datetime


# Set up logging
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)

# Get the current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# Log file path
log_filename = f"chunk_data_log_{current_datetime}.txt"
log_file_path = os.path.join(log_directory, log_filename)

# Set up logging
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)


# Function to check if file exists
def check_file_exists(file_path):
    """Check if a file exists at the given path."""
    logging.info(f"Checking if file exists: {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"File does not exist: {file_path}")
        return False
    logging.info(f"File exists: {file_path}")
    return True

# Function to convert audio to one channel
def convert_to_one_channel(audio_path):
    """Convert audio to one channel (mono)."""
    logging.info(f"Converting audio to one channel: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_channels(1)
    return audio

# Function to parse SRT file
def parse_srt(file_path):
    """Parse an SRT file and return a list of chunks with start time, end time, and text."""
    logging.info(f"Parsing SRT file: {file_path}")
    subs = pysrt.open(file_path)
    chunks = [(sub.start.ordinal, sub.end.ordinal, sub.text) for sub in subs]
    return chunks

# Function to parse VTT file
def parse_vtt(file_path):
    """Parse a VTT file and return a list of chunks with start time, end time, and text."""
    logging.info(f"Parsing VTT file: {file_path}")
    subs = webvtt.read(file_path)
    chunks = [(cue.start_in_seconds * 1000, cue.end_in_seconds * 1000, cue.text) for cue in subs]
    return chunks

# Function to save audio chunks and transcripts
def save_chunks(audio, chunks, base_audio_path, base_transcript_path, output_media_dir, output_transcript_dir):
    """Save audio chunks and transcripts to the output directories."""
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

# Main processing function
def process_files(excel_path, base_audio_path, base_transcript_path, output_media_dir, output_transcript_dir):
    """Process files listed in the Excel sheet."""
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

def main():
    try:
        
        excel_path = 'test.xlsx'
        base_audio_path = 'data/test/english/media'
        base_transcript_path = 'data/test/english/transcriptions'
        output_media_dir = 'data/test-chunk/english/media'
        output_transcript_dir = 'data/test-chunk/english/transcriptions'


        os.makedirs(output_media_dir, exist_ok=True)
        os.makedirs(output_transcript_dir, exist_ok=True)

        logging.info("Starting the file processing")
        process_files(excel_path, base_audio_path, base_transcript_path, output_media_dir, output_transcript_dir)
        logging.info("File processing completed")

    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")

if __name__ == '__main__':
    main()