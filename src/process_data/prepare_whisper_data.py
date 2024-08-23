import argparse
from datasets import Dataset, Audio, Value

def parse_arguments():
    """
    Parse command line arguments.

    Returns:
    - argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Preliminary data preparation script before Whisper Fine-tuning.')
    parser.add_argument('--source_audio_path', type=str, required=True, 
                        help='Path to the file containing the paths to the audio files.')
    parser.add_argument('--source_transcription_path', type=str, required=True, 
                        help='Path to the file containing the transcriptions.')
    parser.add_argument('--output_data_dir', type=str, required=False, default='op_data_dir', 
                        help='Output data directory path.')
    parser.add_argument('--language', type=str, required=True, 
                        help='Language label to add to the dataset.')
    return parser.parse_args()

def load_entries(file_path):
    """
    Load entries from a text file.

    Parameters:
    - file_path (str): Path to the text file.

    Returns:
    - list: List of lines from the file.
    """
    with open(file_path, 'r') as file:
        return file.readlines()

def create_dataset(audio_paths, transcriptions, language):
    """
    Create a Hugging Face Dataset from audio paths and transcriptions.

    Parameters:
    - audio_paths (list): List of paths to audio files.
    - transcriptions (list): List of transcription texts.
    - language (str): Language label to add to the dataset.

    Returns:
    - Dataset: A Hugging Face Dataset object.
    """
    audio_dataset = Dataset.from_dict({
        "audio": [audio_path.strip() for audio_path in audio_paths],
        "sentence": [text_line.strip() for text_line in transcriptions]
    })

    audio_dataset = audio_dataset.cast_column("audio", Audio(sampling_rate=16_000))
    audio_dataset = audio_dataset.cast_column("sentence", Value("string"))

    # Add language column
    language_column = [language] * len(audio_dataset)
    audio_dataset = audio_dataset.add_column("language", language_column)
    
    return audio_dataset

def save_dataset(dataset, output_dir):
    """
    Save the dataset to disk.

    Parameters:
    - dataset (Dataset): The Hugging Face Dataset to save.
    - output_dir (str): Directory where the dataset will be saved.

    Returns:
    - None
    """
    dataset.save_to_disk(output_dir)
    print('Data preparation done')

def main():
    """
    Main function to run the data preparation script.
    
    It loads audio paths and transcriptions, creates a dataset, and saves it to disk.
    
    Returns:
    - None
    """
    args = parse_arguments()

    scp_entries = load_entries(args.source_audio_path)
    txt_entries = load_entries(args.source_transcription_path)

    if len(scp_entries) != len(txt_entries):
        print('Please re-check the audio_paths and text files. They seem to have a mismatch in terms of the number of entries. Both these files should be carrying the same number of lines.')
        return

    dataset = create_dataset(scp_entries, txt_entries, args.language)
    save_dataset(dataset, args.output_data_dir)

if __name__ == '__main__':
    main()


# python prepare_whisper_data.py --source_audio_path path/to/audio_paths.txt \
#                                --source_transcription_path path/to/transcriptions.txt \
#                                --output_data_dir path/to/output/directory \
#                                --language en