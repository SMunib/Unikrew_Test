from pathlib import Path
from modules.Preprocessing import CreateCleanDataset

base_path = Path("dataset")
img_path = Path("img")
entities_path = Path("entities")
bbox_path = Path("box")
output_dir = Path("processed_data")

create_clean_dataset = CreateCleanDataset(base_path,bbox_path,entities_path,img_path,output_dir)
train_data = create_clean_dataset.create_dataset("train")
test_data = create_clean_dataset.create_dataset("test")
# val_data = create_clean_dataset.create_dataset("val")
# print(val_data)
# print(len(val_data))
create_clean_dataset.write_dataset(train_data,"train","train")
create_clean_dataset.write_dataset(test_data,"test","test")
# create_clean_dataset.write_dataset(val_data,"val","val")