import os
import shutil
import pandas as pd

def copy_matching_audio(folder1, folder2, output_folder):
    """
    Copies files from folder2 to output_folder if the part of the filename
    before the first underscore is present in the filenames from folder1.
    """
    os.makedirs(output_folder, exist_ok=True)

    names_set = set()
    for filename in os.listdir(folder1):
        part_name = filename.split('_audio')[0]
        names_set.add(part_name)

    for filename in os.listdir(folder2):
        part_name = filename.split('_audio')[0]
        if part_name in names_set:
            src = os.path.join(folder2, filename)
            dst = os.path.join(output_folder, filename)
            shutil.copy2(src, dst)
            print(f"Copied file: {filename}")

def create_mbti_csv(audio_folder, csv_path, output_csv):
    df = pd.read_csv(csv_path)

    # Extract base names from audio files and convert to lowercase
    audio_names = set()
    for filename in os.listdir(audio_folder):
        part_name = filename.split('_audio')[0].lower()
        audio_names.add(part_name)

    # Convert the CSV column 'username' to lowercase for matching
    df['username_lower'] = df['username'].str.lower()

    matched_data = df[df['username_lower'].isin(audio_names)]

    # Drop the helper column before saving
    matched_data = matched_data.drop(columns=['username_lower'])

    matched_data.to_csv(output_csv, index=False)
    print(f"Created MBTI CSV: {output_csv}")

# Example usage:

# To copy files:
# copy_matching_audio("path/to/first_folder", "path/to/second_folder", "path/to/output_folder")

# To create MBTI CSV separately:
create_mbti_csv("/Users/margotiamanova/Desktop/PROJECTS/smart_audio_trim/output_audio", "/Users/margotiamanova/Desktop/PROJECTS/smart_audio_trim/merged_user_mbti_all.csv", "matched_mbti.csv")
