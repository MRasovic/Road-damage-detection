import os
import pathlib
import DataFormatting

directory = r"C:\Users\Korisnik\Desktop\whole_dataset\pooled_data\labels"
dir_path = r"C:\Users\Korisnik\Desktop\whole_dataset\pooled_data"
# DataFormatting.confirm_deletion(r"C:\Users\Korisnik\RDD\Data_YOLOv8txt_formatted",".txt")

#path_to_dir = Path(r'C:\Users\Korisnik\Desktop\formatted_jp_in_cz_dataset\train')

#for path in path_to_dir.rglob("*"):

#DataFormatting.label_count_info(dir_path)
input_dir_cz = r'C:\Users\Korisnik\RDD\Data_YOLOv8txt_formatted\Cz\OriginalData\annotations'
output_combined_dir = r'C:\Users\Korisnik\Desktop\whole_dataset\pooled_data\labels'


# remove_null_data(directory,400)

DataFormatting.label_count_info(directory)
# DataFormatting.remove_null_data(dir_path,800)
# DataFormatting.label_count_info(directory)