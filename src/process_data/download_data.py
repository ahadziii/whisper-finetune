import os
import logging
import pandas as pd
from google.cloud import storage
from moviepy.editor import VideoFileClip
from datetime import datetime
import argparse

# Initialize the Google Cloud Storage client
client = storage.Client()

def setup_logging(log_directory):
    """
    Set up logging to a file and console with a timestamp.

    Args:
    - log_directory (str): Directory to store log files.

    Returns:
    - str: Path to the log file.
    """
    os.makedirs(log_directory, exist_ok=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_filename = f"download_log_{current_datetime}.txt"
    log_file_path = os.path.join(log_directory, log_filename)

    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return log_file_path

def download_blob(bucket, blob_name, destination_file_name):
    """
    Download a blob from the bucket to a local file.

    Args:
    - bucket (storage.Bucket): Google Cloud Storage bucket.
    - blob_name (str): The name of the blob in the bucket.
    - destination_file_name (str): The local file path where the blob will be saved.

    Returns:
    - None
    """
    try:
        blob = bucket.blob(blob_name)
        blob.download_to_filename(destination_file_name)
        logging.info(f'Blob {blob_name} downloaded to {destination_file_name}.')
    except Exception as e:
        logging.error(f'Error downloading blob {blob_name}: {e}')

def convert_video_to_audio(video_path, audio_path):
    """
    Convert a video file to an audio file.

    Args:
    - video_path (str): Path to the video file.
    - audio_path (str): Path where the audio file will be saved.

    Returns:
    - None
    """
    try:
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path)
        audio_clip.close()
        video_clip.close()
        logging.info(f'Converted {video_path} to {audio_path}.')
    except Exception as e:
        logging.error(f'Error converting video {video_path} to audio: {e}')

def read_excel(file_path):
    """
    Read an Excel file into a pandas DataFrame.

    Args:
    - file_path (str): Path to the Excel file.

    Returns:
    - pd.DataFrame: DataFrame containing the Excel file data.
    """
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        logging.error(f'Error reading Excel file {file_path}: {e}')
        return pd.DataFrame()

def should_exclude_blob_name(blob_name, excluded_emails, excluded_domains):
    """
    Check if the blob name should be excluded based on the given emails and domains.

    Args:
    - blob_name (str): The name of the blob.
    - excluded_emails (list): List of emails to exclude.
    - excluded_domains (list): List of domains to exclude.

    Returns:
    - bool: True if the blob should be excluded, False otherwise.
    """
    for email in excluded_emails:
        if email in blob_name:
            return True
    for domain in excluded_domains:
        if domain in blob_name:
            return True
    return False

def get_blob_name_by_file_name(blobs, file_name):
    """
    Find a blob name by matching the file name.

    Args:
    - blobs (list): List of blobs in the bucket.
    - file_name (str): The file name to search for.

    Returns:
    - str or None: The name of the blob if found, otherwise None.
    """
    for blob in blobs:
        if blob.name.endswith(file_name):
            return blob.name
    logging.warning(f'No blob with the name {file_name} found.')
    return None

def process_files_from_excel(df, bucket, blobs, media_directory, transcription_directory, video_extensions, excluded_emails, excluded_domains):
    """
    Process the files listed in the Excel sheet.

    Args:
    - df (pd.DataFrame): DataFrame containing the file information from Excel.
    - bucket (storage.Bucket): Google Cloud Storage bucket.
    - blobs (list): List of blobs in the bucket.
    - media_directory (str): Directory to save media files.
    - transcription_directory (str): Directory to save transcription files.
    - video_extensions (tuple): Tuple of video file extensions to consider.
    - excluded_emails (list): List of emails to exclude.
    - excluded_domains (list): List of domains to exclude.

    Returns:
    - None
    """
    for index, row in df.iterrows():
        media_file = row['Audio/Video']
        transcript_file = row['Transcript']
        
        if pd.isna(media_file) or pd.isna(transcript_file):
            continue

        media_blob_name = get_blob_name_by_file_name(blobs, media_file)
        transcript_blob_name = get_blob_name_by_file_name(blobs, transcript_file)
        
        if not media_blob_name or not transcript_blob_name:
            logging.warning(f'Missing media or transcript file for media: {media_file}, transcript: {transcript_file}. Skipping...')
            continue
        
        if should_exclude_blob_name(media_blob_name, excluded_emails, excluded_domains) or \
           should_exclude_blob_name(transcript_blob_name, excluded_emails, excluded_domains):
            logging.info(f'Skipping {media_blob_name} and {transcript_blob_name} due to exclusion list...')
            continue
        
        if audio_already_exists(media_blob_name, media_directory):
            logging.info(f'Audio file for {media_blob_name} already exists, skipping download and conversion.')
            continue
        
        media_file_path = os.path.join(media_directory, os.path.basename(media_blob_name))
        os.makedirs(os.path.dirname(media_file_path), exist_ok=True)
        logging.info(f'Downloading media file {media_blob_name}')
        download_blob(bucket, media_blob_name, media_file_path)
        
        if media_blob_name.endswith(video_extensions):
            audio_file_path = media_file_path.rsplit('.', 1)[0] + '.wav'
            logging.info(f'Converting video {media_file_path} to audio {audio_file_path}')
            convert_video_to_audio(media_file_path, audio_file_path)

            logging.info(f'Removing video file {media_file_path}')
            os.remove(media_file_path)
        
        transcription_file_path = os.path.join(transcription_directory, os.path.basename(transcript_blob_name))
        os.makedirs(os.path.dirname(transcription_file_path), exist_ok=True)
        logging.info(f'Downloading transcription file {transcript_blob_name}')
        download_blob(bucket, transcript_blob_name, transcription_file_path)

def audio_already_exists(video_blob_name, media_directory):
    """
    Check if the audio file already exists for the given video blob.

    Args:
    - video_blob_name (str): The name of the video blob.
    - media_directory (str): Directory to check for existing audio files.

    Returns:
    - bool: True if the audio file exists, False otherwise.
    """
    audio_file_name = os.path.basename(video_blob_name).rsplit('.', 1)[0] + '.wav'
    audio_file_path = os.path.join(media_directory, audio_file_name)
    return os.path.exists(audio_file_path)

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
    - argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process media files and their transcriptions.')
    parser.add_argument('--bucket_name', required=True, help='Google Cloud Storage bucket name')
    parser.add_argument('--excel_file_path', required=True, help='Path to the Excel file')
    parser.add_argument('--media_directory', required=True, help='Directory to save media files')
    parser.add_argument('--transcription_directory', required=True, help='Directory to save transcription files')
    parser.add_argument('--log_directory', required=True, help='Directory to save log files')
    parser.add_argument('--excluded_emails', nargs='*', default=[], help='List of emails to exclude')
    parser.add_argument('--excluded_domains', nargs='*', default=[], help='List of domains to exclude')
    return parser.parse_args()

def main():
    """
    Main function to execute the processing of media and transcription files.

    This function parses command-line arguments, sets up logging, and processes the media and transcription files.

    Returns:
    - None
    """
    args = parse_arguments()

    log_file_path = setup_logging(args.log_directory)
    logging.info(f"Logging to {log_file_path}")

    try:
        bucket = client.get_bucket(args.bucket_name)
        
        blobs = list(bucket.list_blobs())

        df = read_excel(args.excel_file_path)
        
        if df.empty:
            logging.error('Excel file is empty or could not be read.')
            return
        
        process_files_from_excel(
            df, bucket, blobs, args.media_directory, args.transcription_directory,
            video_extensions=('.mp4', '.avi', '.mov', '.m4v'),
            excluded_emails=args.excluded_emails,
            excluded_domains=args.excluded_domains
        )
    except Exception as e:
        logging.error(f'Error in main process: {e}')

if __name__ == "__main__":
    main()


    # python script.py \
    # --bucket_name my-gcs-bucket \
    # --excel_file_path /home/user/data/file_list.xlsx \
    # --media_directory /home/user/data/media \
    # --transcription_directory /home/user/data/transcriptions \
    # --log_directory /home/user/data/logs \
    # --excluded_emails test@example.com another@example.com \
    # --excluded_domains example.com test.com