import os
import logging
import pandas as pd
from google.cloud import storage
from moviepy.editor import VideoFileClip
from datetime import datetime

# Initialize the Google Cloud Storage client
client = storage.Client()

# Your bucket name
bucket_name = 'spokentuntikirjanpito.appspot.com'

# Domains and individual emails to exclude
excluded_domains = [
    "semantix.com",
    "acolad.com",
    "ely-keskus.fi",
    "te-toimisto.fi",
    "evl.fi",
    "helsinki.fi",
    "skane.se",
    "lund.se",
    "pirha.fi",
    "forssa.fi",
    "ttl.fi",
    "roihulaw.fi",
]

excluded_emails = [
    "merja.anis@utu.fi",
    "joni.krekola@eduskunta.fi",
    "katri.pardon@helsinki.fi",
    "aino.heikkila@e2.fi",
    "tarja.lahdemaki@valoa.io",
    "timo.a.m.aho@jyu.fi",
    "saila.pesonen@forcit.fi",
    "antti.tolppanen@aalto.fi",
    "toimisto@mannistopalmula.fi",
    "sanna.korpela@xamk.fi",
    "keha@eracontent.com",
    "whisper",
]

# File extensions to download
video_extensions = ('.mp4', '.avi', '.mov', '.m4v')

# Local directory to save files
local_directory = 'data'
log_directory = 'logs'
media_directory = os.path.join(local_directory, 'media')
transcription_directory = os.path.join(local_directory, 'transcriptions')

# Ensure local directories exist
os.makedirs(media_directory, exist_ok=True)
os.makedirs(transcription_directory, exist_ok=True)
os.makedirs(log_directory, exist_ok=True)

# Get the current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# Log file path
log_filename = f"download_log_{current_datetime}.txt"
log_file_path = os.path.join(log_directory, log_filename)

# Set up logging
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def download_blob(bucket, blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    try:
        blob = bucket.blob(blob_name)
        blob.download_to_filename(destination_file_name)
        logging.info(f'Blob {blob_name} downloaded to {destination_file_name}.')
    except Exception as e:
        logging.error(f'Error downloading blob {blob_name}: {e}')

def convert_video_to_audio(video_path, audio_path):
    """Converts a video file to an audio file."""
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
    """Reads the Excel file and returns a DataFrame."""
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        logging.error(f'Error reading Excel file {file_path}: {e}')
        return pd.DataFrame()

def should_exclude_blob_name(blob_name):
    """Checks if the blob name contains any of the excluded emails or domains."""
    for email in excluded_emails:
        if email in blob_name:
            return True
    for domain in excluded_domains:
        if domain in blob_name:
            return True
    return False

def get_blob_name_by_file_name(blobs, file_name):
    """Finds a blob name by matching the file name."""
    for blob in blobs:
        if blob.name.endswith(file_name):
            return blob.name
    logging.warning(f'No blob with the name {file_name} found.')
    return None

def process_files_from_excel(df, bucket, blobs):
    """Processes the files listed in the Excel sheet."""
    for index, row in df.iterrows():
        media_file = row['Audio/Video']
        transcript_file = row['Transcript']
        
        # Skip rows with missing media or transcript file names
        if pd.isna(media_file) or pd.isna(transcript_file):
            continue

        media_blob_name = get_blob_name_by_file_name(blobs, media_file)
        transcript_blob_name = get_blob_name_by_file_name(blobs, transcript_file)
        
        # Check if both the media and transcript files exist
        if not media_blob_name or not transcript_blob_name:
            logging.warning(f'Missing media or transcript file for media: {media_file}, transcript: {transcript_file}. Skipping...')
            continue
        
        # Check if the blob name should be excluded
        if should_exclude_blob_name(media_blob_name) or should_exclude_blob_name(transcript_blob_name):
            logging.info(f'Skipping {media_blob_name} and {transcript_blob_name} due to exclusion list...')
            continue
        
        # Check if audio file already exists
        if audio_already_exists(media_blob_name):
            logging.info(f'Audio file for {media_blob_name} already exists, skipping download and conversion.')
            continue
        
        # Download media file
        media_file_path = os.path.join(media_directory, os.path.basename(media_blob_name))
        os.makedirs(os.path.dirname(media_file_path), exist_ok=True)
        logging.info(f'Downloading media file {media_blob_name}')
        download_blob(bucket, media_blob_name, media_file_path)
        
        if media_blob_name.endswith(video_extensions):
            # Convert video to audio
            audio_file_path = media_file_path.rsplit('.', 1)[0] + '.wav'
            logging.info(f'Converting video {media_file_path} to audio {audio_file_path}')
            convert_video_to_audio(media_file_path, audio_file_path)

            # Remove the original video file
            logging.info(f'Removing video file {media_file_path}')
            os.remove(media_file_path)
            
        
        # Download corresponding transcription file
        transcription_file_path = os.path.join(transcription_directory, os.path.basename(transcript_blob_name))
        os.makedirs(os.path.dirname(transcription_file_path), exist_ok=True)
        logging.info(f'Downloading transcription file {transcript_blob_name}')
        download_blob(bucket, transcript_blob_name, transcription_file_path)

def audio_already_exists(video_blob_name):
    audio_file_name = os.path.basename(video_blob_name).rsplit('.', 1)[0] + '.wav'
    audio_file_path = os.path.join(media_directory, audio_file_name)
    return os.path.exists(audio_file_path)

def main():
    try:
        bucket = client.get_bucket(bucket_name)
        
        # List all blobs in the bucket
        blobs = list(bucket.list_blobs())

        # Read the Excel file
        excel_file_path = 'norway-train-2.xlsx'
        df = read_excel(excel_file_path)
        
        if df.empty:
            logging.error('Excel file is empty or could not be read.')
            return
        
        # Process files listed in the Excel sheet
        process_files_from_excel(df, bucket, blobs)
    except Exception as e:
        logging.error(f'Error in main process: {e}')

if __name__ == "__main__":
    main()
