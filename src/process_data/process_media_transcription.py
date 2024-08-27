import os
import argparse
import urllib.parse
import logging
from datetime import datetime

def setup_logging():
    """
    Set up logging configuration.

    Creates a log file with a timestamp and sets the logging level and format.
    
    Returns:
    - None
    """
    log_directory = 'logs'
    os.makedirs(log_directory, exist_ok=True)

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_filename = f"process_media_transcriptions_log_{current_datetime}.txt"
    log_file_path = os.path.join(log_directory, log_filename)

    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

def create_directories_for_output(media_output_file, transcription_output_file):
    """
    Create directories for output files if they do not already exist.

    Parameters:
    - media_output_file (str): Path where the media output file will be saved.
    - transcription_output_file (str): Path where the transcription output file will be saved.

    Returns:
    - None
    """
    logging.info("Creating directories for output files if they don't exist.")
    os.makedirs(os.path.dirname(media_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(transcription_output_file), exist_ok=True)

def process_media_and_transcriptions(media_dir, transcription_dir, media_output_file, transcription_output_file):
    """
    Process media files and their corresponding transcription files.

    This function iterates over all media files in the specified directory, finds matching transcription files, 
    and writes the paths of the media files and the content of the transcription files to the respective output files.

    Parameters:
    - media_dir (str): Path to the directory containing media files.
    - transcription_dir (str): Path to the directory containing transcription files.
    - media_output_file (str): Path where the processed media file paths will be saved.
    - transcription_output_file (str): Path where the processed transcription content will be saved.

    Returns:
    - None
    """
    create_directories_for_output(media_output_file, transcription_output_file)
    
    with open(media_output_file, 'w') as media_file, open(transcription_output_file, 'w') as transcription_file:
        for media_filename in os.listdir(media_dir):
            media_path = os.path.join(media_dir, media_filename)
            
            if os.path.isfile(media_path):
                try:
                    logging.info(f"Processing media file: {media_filename}")
                    media_file.write(media_path + '\n')
                    media_basename = os.path.splitext(media_filename)[0]
                    transcription_filename = find_transcription_file(transcription_dir, media_basename)
                    
                    if transcription_filename:
                        transcription_path = os.path.join(transcription_dir, transcription_filename)
                        logging.info(f"Found transcription file: {transcription_filename} for media file: {media_filename}")
                        
                        with open(transcription_path, 'r') as tf:
                            transcription_content = tf.read().replace('\n', ' ')
                        
                        transcription_file.write(transcription_content + '\n')
                    else:
                        logging.warning(f"Transcription file not found for media: {media_filename}")
                except Exception as e:
                    logging.error(f"Error processing file {media_filename}: {e}")

def find_transcription_file(transcription_dir, media_basename):
    """
    Find a transcription file that matches the base name of the media file.

    This function searches the transcription directory for a file that has the same base name as the media file.

    Parameters:
    - transcription_dir (str): Path to the directory containing transcription files.
    - media_basename (str): Base name of the media file.

    Returns:
    - str or None: The filename of the matching transcription file, or None if no match is found.
    """
    media_basename_decoded = urllib.parse.unquote(media_basename)
    logging.info(f"Looking for transcription file matching base name: {media_basename_decoded}")
    
    for filename in os.listdir(transcription_dir):
        filename_decoded = urllib.parse.unquote(filename)
        
        if media_basename_decoded == os.path.splitext(filename_decoded)[0]:
            logging.info(f"Match found: {filename} for base name: {media_basename_decoded}")
            return filename
    
    logging.warning(f"No match found for base name: {media_basename_decoded}")
    return None

def parse_arguments():
    """
    Parse command-line arguments using argparse.

    Returns:
    - argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Process media files and their transcriptions.')
    parser.add_argument('--media_directory', required=True, help='Path to the directory containing media files')
    parser.add_argument('--transcription_directory', required=True, help='Path to the directory containing transcription files')
    parser.add_argument('--output', required=True, help='Path to the output directory')
    parser.add_argument('--language', type=str, required=True, help='Language of media')

    return parser.parse_args()

def main():
    """
    Main function to execute the processing of media and transcription files.

    This function parses command-line arguments, determines the output paths, and processes the media and transcription files.

    Returns:
    - None
    """
    setup_logging()
    try:
        args = parse_arguments()

        media_output_file = os.path.join(args.output, f'media_{args.language}.txt')
        transcription_output_file = os.path.join(args.output, f'transcription_{args.language}.txt')

        logging.info(f"Starting process with media directory: {args.media_directory}, transcription directory: {args.transcription_directory}, output directory: {args.output}, language: {args.language}")
        process_media_and_transcriptions(args.media_directory, args.transcription_directory, media_output_file, transcription_output_file)
        logging.info("Processing complete.")
    
    except Exception as e:
        logging.error(f"Error in main process: {e}")

if __name__ == "__main__":
    main()


# python process_media_transcriptions.py --media_directory /path/to/media_dir \
#                                        --transcription_directory /path/to/transcription_dir \
#                                        --output /path/to/output_dir \
#                                        --language en