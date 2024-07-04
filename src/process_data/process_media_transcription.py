import os
import argparse
import urllib.parse

def process_media_and_transcriptions(media_dir, transcription_dir, media_output_file, transcription_output_file):
    print(f"Creating directories for output files if they don't exist.")
    os.makedirs(os.path.dirname(media_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(transcription_output_file), exist_ok=True)
    
    with open(media_output_file, 'w') as media_file, open(transcription_output_file, 'w') as transcription_file:
        for media_filename in os.listdir(media_dir):
            media_path = os.path.join(media_dir, media_filename)
            
            if os.path.isfile(media_path):
                try:
                    print(f"Processing media file: {media_filename}")
                    media_file.write(media_path + '\n')
                    media_basename = os.path.splitext(media_filename)[0]
                    transcription_filename = find_transcription_file(transcription_dir, media_basename)
                    
                    if transcription_filename:
                        transcription_path = os.path.join(transcription_dir, transcription_filename)
                        print(f"Found transcription file: {transcription_filename} for media file: {media_filename}")
                        
                        with open(transcription_path, 'r') as tf:
                            transcription_content = tf.read().replace('\n', ' ')
                        
                        transcription_file.write(transcription_content + '\n')
                    else:
                        print(f"Transcription file not found for media: {media_filename}")
                except Exception as e:
                    print(f"Error processing file {media_filename}: {e}")

def find_transcription_file(transcription_dir, media_basename):
    media_basename_decoded = urllib.parse.unquote(media_basename)
    print(f"Looking for transcription file matching base name: {media_basename_decoded}")
    
    for filename in os.listdir(transcription_dir):
        filename_decoded = urllib.parse.unquote(filename)
        
        if media_basename_decoded == os.path.splitext(filename_decoded)[0]:
            print(f"Match found: {filename} for base name: {media_basename_decoded}")
            return filename
    
    print(f"No match found for base name: {media_basename_decoded}")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process media files and their transcriptions.')
    parser.add_argument('--media_directory', required=True, help='Path to the directory containing media files')
    parser.add_argument('--transcription_directory', required=True, help='Path to the directory containing transcription files')
    parser.add_argument('--output', required=True, help='Path to the output directory')
    parser.add_argument('--language', type=str, required=True, help='Language of media')

    args = parser.parse_args()

    media_output_file = os.path.join(args.output, f'media_{args.language}.txt')
    transcription_output_file = os.path.join(args.output, f'transcription_{args.language}.txt')

    print(f"Starting process with media directory: {args.media_directory}, transcription directory: {args.transcription_directory}, output directory: {args.output}, language: {args.language}")
    process_media_and_transcriptions(args.media_directory, args.transcription_directory, media_output_file, transcription_output_file)
    print("Processing complete.")