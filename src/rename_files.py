import os

def rename_files(folder_path):
    """Rename all files in a folder to a 3-digit number."""
    files = os.listdir(folder_path)
    for index, filename in enumerate(files):
        file_extension = os.path.splitext(filename)[1]
        new_name = f"{index:03d}{file_extension}"
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_name)
        os.rename(old_file, new_file)
        print(f"Renamed '{filename}' to '{new_name}'")

if __name__ == "__main__":
    folder_path = 'data'
    rename_files(folder_path)