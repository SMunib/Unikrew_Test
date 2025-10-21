from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
from difflib import SequenceMatcher
from time import perf_counter
from tqdm import tqdm
from PIL import Image

class CreateCleanDataset(Dataset):
    def __init__(self, base_path:Path,bbox_path:Path,entities_path:Path,img_path:Path,output_dir:Path):

        self.base_path = base_path
        self.bbox_path = bbox_path 
        self.entities_path = entities_path
        self.img_path = img_path
        self.output_dir = output_dir

    def read_bbox_and_words(self, path:Path):

        bbox_and_words_list = []
        with open(path,'r',errors='ignore') as f:
            for line in f.read().splitlines():
                if len(line) == 0:
                    continue

                split_lines = line.split(",")

                bbox = np.array(split_lines[0:8]).astype(np.int32)
                text = ",".join(split_lines[8:])

                bbox_and_words_list.append([path.stem + ".jpg",*bbox,text])

        df = pd.DataFrame(bbox_and_words_list,columns=['filename','x0','y0','x1','y1','x2','y2','x3','y3','line'])
        df =  df.drop(columns=['x1','y1','x3','y3'])
        return df

    def read_entities(self,path:Path):
        
        with open(path,'r') as f:
            data = json.load(f)

        df = pd.DataFrame([data])
        return df    
    
    def assign_line_label(self,line:str,entities: pd.DataFrame):
        
        line_set = line.replace(" ","").strip().split()
        best_label = 'O'
        best_score = 0
        match_reason = "no match"

        for i, column in enumerate(entities.columns):
            entity_text = entities.iloc[0,i].replace(",","").strip()
            entity_set = entity_text.split()

            matches_count = 0
            for word in line_set:
                if any (SequenceMatcher(a=word,b=entity_word).ratio() >= 0.8 for entity_word in entity_set):
                    matches_count += 1
            
            line_match_ratio = matches_count / len(line_set) if line_set else 0
            entity_match_ratio = matches_count / len(entity_set) if entity_set else 0

            if column.upper() == 'ADDRESS' and line_match_ratio > 0.5:
                best_label = column.upper()
                best_score = line_match_ratio
                match_reason = "address ratio >= 0.5"
                break

            elif column.upper() != 'ADDRESS' and matches_count == len(line_set):
                best_label = column.upper()
                best_score = 1.0
                match_reason = "full line match"
                break

            elif matches_count == len(entity_set):
                best_label = column.upper()
                best_score = entity_match_ratio
                match_reason = "full entity match"
                break

        log = {
            "line": line,
            "assigned_label": best_label,
            "confidence_score": round(best_score, 2),
            "match_reason": match_reason
        }

        return best_label, log
    
    def assign_labels(self,words: pd.DataFrame, entities: pd.DataFrame):

        max_area = {"TOTAL": (0, -1), "DATE": (0, -1)}
        already_labeled = {
            "TOTAL": False,
            "DATE": False,
            "ADDRESS": False,
            "COMPANY": False,
        }

        labels = []
        
        for i, line in enumerate(words['line']):
            label,_ = self.assign_line_label(line, entities)

            if label == 'ADDRESS' and already_labeled['TOTAL']:
                label = "O"

            if label == 'COMPANY' and (already_labeled["DATE"] or already_labeled["TOTAL"]):
                label = "O"
            
            if label in ['TOTAL', 'DATE']:
                x0_loc = words.columns.get_loc("x0")
                bbox = words.iloc[i,x0_loc:x0_loc+4].to_list()
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                if area > max_area[label][0]:
                    max_area[label] = (area, i)

                label = 'O'

            else:
                if label != "O":
                    already_labeled[label] = True

            labels.append(label)
        
        if max_area["DATE"][1] != -1:
            labels[max_area["DATE"][1]] = "DATE"
        if max_area["TOTAL"][1] != -1:
            labels[max_area["TOTAL"][1]] = "TOTAL"
        
        words["label"] = labels
        return words
        
    def split_line(self,line: pd.Series):
        
        line_copy = line.copy()
        line_str = line_copy.loc['line']
        words = line_str.split(" ")

        words = [word for word in words if len(word) >= 1]

        x0,y0,x2,y2 = line_copy.loc[['x0','y0','x2','y2']]
        bbox_width = x2 - x0

        new_lines = []
        for index, word in enumerate(words):
            x2 = x0 + int(bbox_width * len(word)/len(line_str))
            line_copy.loc[['x0','x2','line']] = [x0,x2,word]
            new_lines.append(line_copy.to_list())
            x0 = x2 + 5
        
        return new_lines
    
    def create_dataset(self,split:str):
        
        base_path = self.base_path / split
        entities_files = sorted((base_path / self.entities_path).glob("*.txt"))
        bbox_files = sorted((base_path / self.bbox_path).glob("*.txt"))
        img_files = sorted((base_path / self.img_path).glob("*.jpg"))

        entities_files = entities_files[:int(0.3 * len(entities_files))]
        bbox_files = bbox_files[:int(0.3 * len(bbox_files))]
        img_files = img_files[:int(0.3 * len(img_files))]

        print(len(entities_files))
        
        data = []
        print("Reading Dataset......")
        for bbox_file, entity_file, img_file in tqdm(zip(bbox_files, entities_files, img_files), total=len(bbox_files)):
            bbox = self.read_bbox_and_words(bbox_file)
            entities = self.read_entities(entity_file)
            img = Image.open(img_file).convert("RGB")
            bbox_labelled = self.assign_labels(bbox, entities)
            del bbox

            new_bbox_1 = []
            for index,row in bbox_labelled.iterrows():
                new_bbox_1 += self.split_line(row)
            new_bbox = pd.DataFrame(new_bbox_1,columns=bbox_labelled.columns)
            del bbox_labelled

            for index,row in new_bbox.iterrows():
                label = row['label']

                if label != "O":
                    entity_values = entities.iloc[0,entities.columns.get_loc(label.lower())]
                    entity_set = entity_values.split()

                    if any(SequenceMatcher(a=row['line'],b=b).ratio() >= 0.7 for b in entity_set):
                        label = "S-" + label
                    else:
                        label = "O"
                
                new_bbox.at[index,'label'] = label
            
            width,height = img.size
            # new_bbox['filename'] = img_file.name + ".jpg"
            # print(new_bbox['filename'])
            data.append([new_bbox,width,height])
        
        return data

    def normalize(self,points:list,width:int,height:int) -> list:
        x0,y0,x2,y2 = [int(p) for p in points]

        x0 = int(1000 * (x0/width))
        y0 = int(1000 * (y0/height))
        x2 = int(1000 * (x2/width))
        y2 = int(1000 * (y2/height))

        return [x0,y0,x2,y2]
    
    def write_dataset(self,dataset:list,name:str,split:str):
        print(f"Writing {name}ing Dataset......")
        (self.output_dir / split).mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / split / f"{name}.txt",'w+',encoding='utf-8') as file, \
             open(self.output_dir / split / f"{name}_box.txt",'w+',encoding='utf-8') as bbox_file, \
             open(self.output_dir / split / f"{name}_img.txt",'w+',encoding='utf-8') as img_file:
            
            for data_item in tqdm(dataset,total=len(dataset)):
                data,width,height = data_item
                # print(data['filename'])

                filename = data.iloc[0,data.columns.get_loc('filename')]
                # print(filename)
                for index, row in data.iterrows():
                    bbox = [int(p) for p in row[['x0','y0','x2','y2']]]
                    normalized_bbox = self.normalize(bbox,width,height)

                    file.write("{}\t{}\n".format(row['line'],row['label']))
                    bbox_file.write("{}\t{} {} {} {}\n".format(row['line'], *normalized_bbox))
                    img_file.write("{}\t{} {} {} {}\t{} {}\t{}\n".format(row['line'],*bbox,width,height,filename,row['filename']))
                    
                file.write("\n")
                bbox_file.write("\n")
                img_file.write("\n")

        print(f"Dataset {name} written successfully in {self.output_dir / split}")