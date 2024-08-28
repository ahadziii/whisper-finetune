import os
import argparse
import urllib.parse

def create_output_directories(media_output_file, transcription_output_file):
    """
    Create directories for the output files if they don't exist.

    Parameters:
    - media_output_file (str): Path where the media output file will be saved.
    - transcription_output_file (str): Path where the transcription output file will be saved.

    Returns:
    - None
    """
    print(f"Creating directories for output files if they don't exist.")
    os.makedirs(os.path.dirname(media_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(transcription_output_file), exist_ok=True)

def write_media_paths(media_dir, media_output_file):
    """
    Write paths of media files to the media output file.

    Parameters:
    - media_dir (str): Path to the directory containing media files.
    - media_output_file (str): Path where the processed media file paths will be saved.

    Returns:
    - None
    """
    with open(media_output_file, 'w') as media_file:
        for media_filename in os.listdir(media_dir):
            media_path = os.path.join(media_dir, media_filename)
            if os.path.isfile(media_path):
                media_file.write(media_path + '\n')

def process_transcriptions(media_dir, transcription_dir, transcription_output_file):
    """
    Process and save transcription content for each media file.

    Parameters:
    - media_dir (str): Path to the directory containing media files.
    - transcription_dir (str): Path to the directory containing transcription files.
    - transcription_output_file (str): Path where the processed transcription content will be saved.

    Returns:
    - None
    """
    with open(transcription_output_file, 'w') as transcription_file:
        for media_filename in os.listdir(media_dir):
            media_basename = os.path.splitext(media_filename)[0]
            transcription_filename = find_transcription_file(transcription_dir, media_basename)

            if transcription_filename:
                transcription_path = os.path.join(transcription_dir, transcription_filename)
                try:
                    transcription_content = read_transcription_file(transcription_path)
                    transcription_file.write(transcription_content + '\n')
                except Exception as e:
                    print(f"Error reading transcription file {transcription_filename}: {e}")
            else:
                print(f"Transcription file not found for media: {media_filename}")

def find_transcription_file(transcription_dir, media_basename):
    """
    Find a transcription file that matches the base name of the media file.

    Parameters:
    - transcription_dir (str): Path to the directory containing transcription files.
    - media_basename (str): Base name of the media file.

    Returns:
    - str or None: The filename of the matching transcription file, or None if no match is found.
    """
    media_basename_decoded = urllib.parse.unquote(media_basename)
    print(f"Looking for transcription file matching base name: {media_basename_decoded}")

    for filename in os.listdir(transcription_dir):
        filename_decoded = urllib.parse.unquote(filename)

        if media_basename_decoded == os.path.splitext(filename_decoded)[0]:
            print(f"Match found: {filename} for base name: {media_basename_decoded}")
            return filename

    print(f"No match found for base name: {media_basename_decoded}")
    return None

def read_transcription_file(transcription_path):
    """
    Read and process the content of a transcription file.

    Parameters:
    - transcription_path (str): Path to the transcription file.

    Returns:
    - str: Processed transcription content.
    """
    with open(transcription_path, 'r') as tf:
        return tf.read().replace('\n', ' ')

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

    This function parses command-line arguments, determines the output paths, creates necessary directories,
    and processes the media and transcription files.

    Returns:
    - None
    """
    args = parse_arguments()

    media_output_file = os.path.join(args.output, f'media_{args.language}.txt')
    transcription_output_file = os.path.join(args.output, f'transcription_{args.language}.txt')

    create_output_directories(media_output_file, transcription_output_file)
    write_media_paths(args.media_directory, media_output_file)
    process_transcriptions(args.media_directory, args.transcription_directory, transcription_output_file)

    print("Processing complete.")

if __name__ == "__main__":
    main()