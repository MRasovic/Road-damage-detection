import os
import shutil
import random
import xml.etree.ElementTree as ET

### Splitting the directory into train test and validation subdirectories by split ratio
### Note that file creation needs to be run first
def create_train_test_split_dirs(destination_dir):

    # Create train, test, and validation directories
    train_dir = os.path.join(destination_dir, 'train')
    test_dir = os.path.join(destination_dir, 'test')
    val_dir = os.path.join(destination_dir, 'validation')

    # Check if said directories exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Populate directories with annotation and image directories
    train_annotation_dir = os.path.join(train_dir, 'labels')
    train_image_dir = os.path.join(train_dir, 'images')

    test_annotation_dir = os.path.join(test_dir, 'labels')
    test_image_dir = os.path.join(test_dir, 'images')

    val_annotation_dir = os.path.join(val_dir, 'labels')
    val_image_dir = os.path.join(val_dir, 'images')

    os.makedirs(train_annotation_dir, exist_ok=True)  # exist_ok means if the target directory already exists it will simply ignore the creation of the directory and move on.
    os.makedirs(train_image_dir, exist_ok=True)

    os.makedirs(test_annotation_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)

    os.makedirs(val_annotation_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
def is_directory_empty(*argv):
    """
    Helper function to check if a directory and its children are empty
    :param argv: Any number of string paths to directories
    :return: Boolean that answers are all directories and its subdirs empty
    """
    for directory in argv:
        if not os.path.isdir(directory):
            print(f"Error: {directory} is not a valid directory.")
            continue
        dir_entries = os.listdir(directory)

        for entry in dir_entries:
            full_path = os.path.join(directory, 'labels')
            if os.path.isdir(full_path):
                if not is_directory_empty(full_path):  # Recursively check subdirectories
                    return False
            else:
                # If any file is found, the directory is not empty
                print(f"Non-empty file found: {full_path}")
                return False

    return True
def split_files(source_dir, train_dir, test_dir ,val_dir ,annot_format , split_ratio=(0.7, 0.2, 0.1)):
    """
    Repopulates train val and test directories with randomly pulled files from a source_dir

    :param annot_format:
    :param source_dir:
    :param train_dir:
    :param test_dir:
    :param val_dir:
    :param split_ratio: train  test val is the tuple, in that order
    """

    # Check for read write and delete permissions in directories
    if not (os.access(source_dir, os.R_OK) and os.access(source_dir, os.W_OK) and os.access(source_dir, os.X_OK)):
        print("Source directory does not have necessary permissions.")
        return

    try:
        if not is_directory_empty(train_dir, test_dir, val_dir):
            print("Directories are already populated. Stopping process...")
            return

        for root, dirs, files in os.walk(source_dir):
            if "images" in dirs:
                images_dir = os.path.join(root, "images")
                file_paths = [os.path.join(images_dir, file) for file in os.listdir(images_dir)]

                for file in file_paths:
                    if file.endswith('.jpg' or ".png"):
                        source_image_file = os.path.join(root, file)

                        destination = random.choices([train_dir, test_dir, val_dir], weights=split_ratio)[0]

                        destination_image_file = os.path.join(destination, 'images', os.path.basename(file))

                        destination_annotation_file = os.path.join(
                            destination, 'labels', os.path.basename(file).replace('.jpg', annot_format))

                        os.makedirs(os.path.dirname(destination_image_file), exist_ok=True)
                        os.makedirs(os.path.dirname(destination_annotation_file), exist_ok=True)

                        shutil.copy2(str(source_image_file), str(destination_image_file))

                        # Adjust the source_annotation_file path
                        source_annotation_file = os.path.join(source_dir, 'labels',
                                                              os.path.basename(file).replace('.jpg', annot_format))

                        shutil.copy2(str(source_annotation_file),
                                     str(destination_annotation_file))  # Copies the annotation file
        else:
            print("Directories are populated, stopping process...")

    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")

### Simple deletion of xml func and a confirmation func
def delete_all_file_type_from_dir(target_dir,file_type):
    """
    :param target_dir:
    :param file_type: (str) file extension
    :return:
    """
    for root,dirs,files in os.walk(target_dir):
        for file in files:
            if file.endswith(file_type):
                file_path = os.path.join(root, file)
                print(f"Deleting item: {file_path}")
                os.remove(file_path)
def confirm_deletion(target_dir,file_type):
    confirm = input(f"Are you sure you want to purge the {target_dir} of all xml files including subdirectories?")

    if confirm.lower() == "y":
        delete_all_file_type_from_dir(target_dir,file_type)
    else:
        print("Canceling deletion")

### Converting VOC xml files to YOLOv8 txt format
def parse_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    objects = []
    for obj in root.findall('object'):
        obj_name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append((obj_name, xmin, ymin, xmax, ymax))

    return objects, root.find('size')
def convert_to_yolo_v8(objects, img_size,class_mapping):
    width = int(img_size.find('width').text)
    height = int(img_size.find('height').text)

    lines = []
    for obj in objects:
        obj_name, xmin, ymin, xmax, ymax = obj
        if obj_name in class_mapping:
            class_index = class_mapping[obj_name]
            center_x = (xmin + xmax) / (2.0 * width)
            center_y = (ymin + ymax) / (2.0 * height)
            obj_width = (xmax - xmin) / width
            obj_height = (ymax - ymin) / height
            lines.append(f"{class_index} {center_x} {center_y} {obj_width} {obj_height}\n")

    return lines
def write_to_yolo_txt(txt_file, lines):
    with open(txt_file, 'w') as f:
        f.writelines(lines)
def convert_annotations_in_directory_from_xml_to_yolo_v8txt(input_dir, output_dir, class_mapping):
    """

    :param input_dir:
    :param output_dir:
    :param class_mapping:
    :return:
    """

    for filename in os.listdir(input_dir):
        print(filename)

        if filename.endswith('.xml'):

            xml_file_path = os.path.join(input_dir, filename)
            objects, img_size = parse_voc_xml(xml_file_path)
            yolo_lines = convert_to_yolo_v8(objects, img_size,class_mapping)
            output_txt_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
            write_to_yolo_txt(output_txt_path, yolo_lines)
def read_annotations_from_xml(xml_file):
    """
    Read annotations from an XML file.

    Args:
        xml_file (str): Path to the XML file containing annotations.

    Returns:
        list: List of dictionaries representing annotations.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_filename = root.find('filename').text
    annotations = []

    # Check for presence of objects
    objects = root.findall("object")

    if len(objects) == 0:  # No bounding box objects found, saving just the filename
        annotations.append({'filename': image_filename, 'name': None})

    else:
        for obj in objects:  # Add all object information to the annotation list
            name = obj.find('name').text  # Name being, the name of the class of road damage
            annotations.append({'filename': image_filename, 'name': name})

    return annotations

### Cleaning datasets of unwanted class annotations
def find_objects_by_name(xml_path, name_to_find):
    """
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find objects with the specified name
    objects_found = []
    for obj in root.findall('object'):
        name_element = obj.find('name')
        print("Looking for objects...")
        if name_to_find == name_element.text:
            objects_found.append(obj)

    print("Size of list:"+" ",len(objects_found))
    return objects_found
def delete_objects(xml_path, objects_to_delete):
    """
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Remove objects from the XML tree
    for obj in objects_to_delete:
        if obj in root:
            root.remove(obj)
            print("Object deleted:", obj)
        else:
            print("Object not found in XML tree:", obj)

    # Write the modified XML back to the file
    tree.write(xml_path)
def delete_annotation_objects_by_name(directory,name):
    """
    """
    # Walking through a directory
    for root_dir,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith('.xml'):
                xml_path = os.path.join(root_dir, file)
                print("Processing XML:", xml_path)
                delete_objects(xml_path,find_objects_by_name(xml_path, name))

def label_count_info(dir_path):
    file_count = 0
    number_counts = {}
    empty_files_count = 0


    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            with open(os.path.join(dir_path, filename), 'r') as file:
                lines = file.readlines()
                if not lines:
                    empty_files_count +=1
                for line in lines:
                    numbers = line.strip().split()
                    if numbers:  # Check if the line is not empty
                        first_number = int(numbers[0])  # Convert first number to integer
                        # adds another pair to dictionary if new, otherwise adds 1 to dict value
                        number_counts[first_number] = number_counts.get(first_number, 0) + 1
                file_count += 1

    print(f"Processed {file_count} files.")
    print(f"Empty files: {empty_files_count}")
    print("Number Counts:")
    for number, count in number_counts.items():
        print(f"{number}: {count}")

def remove_null_data(source_dir,delete_count):
    """
    Recursively enters subdirectories, source dir should be a root directory
    :param source_dir:
    :param delete_count: number of annotation/image pairs to be deleted
    :return: Nan
    """
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.txt') and os.path.getsize(os.path.join(root, file)) == 0 and delete_count != 0:
                txt_to_delete = os.path.join(root, file)
                image_to_delete = os.path.join(source_dir, 'images', file.replace('.txt', ".jpg"))
                try:
                    os.remove(txt_to_delete)
                    os.remove(image_to_delete)
                    print(f"Annotation file '{txt_to_delete}' deleted successfully, along with its '{image_to_delete}' jpg counterpart")
                    delete_count -=1
                except OSError as e:
                    print(f"Error: {e.filename} - {e.strerror}")

def is_empty(folder_path):
    return len(os.listdir(folder_path)) == 0
def copy_images_to_dir(source_dir,target_dir):

    if not os.path.exists(target_dir):
        print("Target directory not found, creating it...")
        os.makedirs(target_dir)
    files = os.listdir(source_dir)

    for file in files:
        source_file_path = os.path.join(source_dir, file)
        if os.path.isfile(source_file_path) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Copy image file to the target directory
            shutil.copy(source_file_path, target_dir)

    print("Images copied successfully.")
