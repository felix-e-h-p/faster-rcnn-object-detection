import fnmatch
import os
import random
import shutil
from PIL import Image

root_name = "frcnn_update"
base_directory = os.path.join("/Users/felixhall/Desktop", root_name)
processed_data_directory = os.path.join(base_directory, "data", "processed")
full_data_directory = os.path.join(processed_data_directory, 'full_data')
os.makedirs(full_data_directory, exist_ok=True)
img_path = os.path.join(processed_data_directory, 'windTurbineDataSet/JPEGImages')
annot_path = os.path.join(processed_data_directory, 'windTurbineDataSet/Annotations')

def create_directories(base_directory, data_subdirectories):
    for subdirectory in data_subdirectories:
        os.makedirs(os.path.join(base_directory, "data", subdirectory), exist_ok=True)

def extract_zip(zip_file, extract_directory, archive_format="zip"):
    try:
        shutil.unpack_archive(zip_file, extract_directory, archive_format)
        print(f"Successfully extracted {zip_file} to {extract_directory}")
    except Exception as e:
        print(f"Error extracting {zip_file}: {e}")

def setup_project_structure(base_directory):
    data_subdirectories = ["raw", "processed"]
    create_directories(base_directory, data_subdirectories)
    zip_file = os.path.join(base_directory, "data", 'raw', 'windTurbineDataSet_xml_annotations.zip')
    extract_zip(zip_file, processed_data_directory)

def move_files_with_annotations(img_path, annot_path, full_data_directory):

    png_files = [f for f in os.listdir(img_path) if f.endswith('.png')]
    for png_file in png_files:
        xml_file = os.path.splitext(png_file)[0] + '.xml'
        if os.path.exists(os.path.join(annot_path, xml_file)):
            shutil.move(os.path.join(img_path, png_file), os.path.join(full_data_directory, png_file))
            shutil.move(os.path.join(annot_path, xml_file), os.path.join(full_data_directory, xml_file))

def corrupt_image_removal(full_data_directory):
    for filename in os.listdir(full_data_directory):
        if filename.endswith('.png'):
            try:
                img_path = os.path.join(full_data_directory, filename)
                img = Image.open(img_path)
                img.verify()
            except (IOError, SyntaxError) as e:
                print(f"Corrupt image: {filename}")
                xml_file = os.path.splitext(filename)[0] + '.xml'
                os.remove(img_path)
                os.remove(os.path.join(full_data_directory, xml_file))
                print(f"Removed {filename} and {xml_file}")

def create_and_move_directories(full_data, base_directory):
    train_dir = os.path.join(base_directory, "data", "training")
    val_dir = os.path.join(base_directory, "data", "validation")
    test_dir = os.path.join(base_directory, "data", "testing")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    png_files = [f for f in os.listdir(full_data_directory) if f.endswith('.png')]
    total_files = len(png_files)
    train_size = int(0.7 * total_files)
    val_size = test_size = int(0.15 * total_files)

    random.shuffle(png_files)

    train_files = png_files[:train_size]
    val_files = png_files[train_size:train_size + val_size]
    test_files = png_files[train_size + val_size:]

    for files, dest_directory in [(train_files, train_dir), (val_files, val_dir), (test_files, test_dir)]:
        for png_file in files:
            xml_file = os.path.splitext(png_file)[0] + '.xml'
            shutil.move(os.path.join(full_data_directory, png_file), os.path.join(dest_directory, png_file))
            shutil.move(os.path.join(full_data_directory, xml_file), os.path.join(dest_directory, xml_file))
    print("Data split complete.")

    count_train = len(fnmatch.filter(os.listdir(train_dir), '*.png'))
    count_val = len(fnmatch.filter(os.listdir(val_dir), '*.png'))
    count_test = len(fnmatch.filter(os.listdir(test_dir), '*.png'))
    
    print('Training images count:', count_train)
    print('Validation images count:', count_val)
    print('Testing images count:', count_test)

setup_project_structure(base_directory)
move_files_with_annotations(img_path, annot_path, full_data_directory)
corrupt_image_removal(full_data_directory)
create_and_move_directories(full_data_directory, base_directory)