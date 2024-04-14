import DataFormatting
import os
from pathlib import Path

if __name__ == '__main__':
    class_mapping= {
        'D00': 0,
        'D10': 1,
        'D20': 2,
        'D40': 3,
        'D44': 4,
        'D50': 5,
        'D43': 6,
    }

    root_dir = r"C:\Users\Korisnik\Desktop\whole_dataset"

    ### Create file structure necessary for training
    #DataFormatting.create_train_test_split_dirs(root_dir)

    ### Formatting to YOLOv8.txt format

    #Input directories
    input_dir_china_drone = r'D:\Work\Road Damage Detection\Datasets\RDD2022_all_countries\China_Drone\train\annotations'
    input_dir_china_motorbike = r'D:\Work\Road Damage Detection\Datasets\RDD2022_all_countries\China_MotorBike\train\annotations'
    input_dir_cz = r'C:\Users\Korisnik\RDD\Data_YOLOv8txt_formatted\Cz\OriginalData\annotations'
    input_dir_in = r'C:\Users\Korisnik\RDD\Data_YOLOv8txt_formatted\In\OriginalData\annotations'
    input_dir_jp = r'C:\Users\Korisnik\RDD\Data_YOLOv8txt_formatted\Jp\OriginalData\annotations'
    input_dir_usa = r"D:\Work\Road Damage Detection\Datasets\RDD2022_all_countries\United_States\train\annotations"

    # Output directories
    output_combined_dir = r'C:\Users\Korisnik\Desktop\whole_dataset\pooled_data\labels'

    ## Czech subset
    # All directories that will combine into the dataset need to be converted from xml to .txt
    DataFormatting.convert_annotations_in_directory_from_xml_to_yolo_v8txt(
        input_dir_cz,
        output_combined_dir,
        class_mapping
     )
    # Copying images from every directory to pooled_data/images dir
    DataFormatting.copy_images_to_dir(
        input_dir_cz.replace("annotations",'images'),
        output_combined_dir.replace("labels",'images')
    )

    ## Indian subset
    DataFormatting.convert_annotations_in_directory_from_xml_to_yolo_v8txt(
        input_dir_in,
        output_combined_dir,
        class_mapping
     )
    DataFormatting.copy_images_to_dir(
        input_dir_in.replace("annotations", 'images'),
        output_combined_dir.replace("labels", 'images')
    )

    ## Japanese subset
    DataFormatting.convert_annotations_in_directory_from_xml_to_yolo_v8txt(
        input_dir_jp,
        output_combined_dir,
        class_mapping
     )
    DataFormatting.copy_images_to_dir(
        input_dir_jp.replace("annotations", 'images'),
        output_combined_dir.replace("labels", 'images')
    )

    ## China drone subset
    DataFormatting.convert_annotations_in_directory_from_xml_to_yolo_v8txt(
        input_dir_china_drone,
        output_combined_dir,
        class_mapping
    )
    DataFormatting.copy_images_to_dir(
        input_dir_china_drone.replace("annotations", 'images'),
        output_combined_dir.replace("labels", 'images')
    )

    ## China bike subset
    DataFormatting.convert_annotations_in_directory_from_xml_to_yolo_v8txt(
        input_dir_china_motorbike,
        output_combined_dir,
        class_mapping
    )
    DataFormatting.copy_images_to_dir(
        input_dir_china_motorbike.replace("annotations", 'images'),
        output_combined_dir.replace("labels", 'images')
    )

    ## USA subset
    DataFormatting.convert_annotations_in_directory_from_xml_to_yolo_v8txt(
        input_dir_usa,
        output_combined_dir,
        class_mapping
    )
    DataFormatting.copy_images_to_dir(
        input_dir_usa.replace("annotations", 'images'),
        output_combined_dir.replace("labels", 'images')
    )

    train_dir = r"C:\Users\Korisnik\Desktop\whole_dataset\test"
    test_dir = r"C:\Users\Korisnik\Desktop\whole_dataset\train"
    val_dir = r"C:\Users\Korisnik\Desktop\whole_dataset\validation"

    combined_dir = r'C:\Users\Korisnik\Desktop\whole_dataset\pooled_data\labels'
    #DataFormatting.split_files(combined_dir, train_dir, test_dir, val_dir, annot_format=".txt")



