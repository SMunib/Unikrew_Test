from torch.utils.data import Dataset
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self,txt_path,img_path,box_path,processor,label2id,img_dir):
        self.words,self.labels,self.bboxes,self.images = self._load_data(txt_path,img_path,box_path,img_dir)
        self.processor = processor
        self.label2id = label2id

    def _load_data(self,txt_path,img_path,box_path,img_dir):
        words_per_doc,labels_per_doc,bboxes_per_doc,images = [],[],[],[]
        with open(txt_path,'r',encoding='utf-8') as f_text,\
             open(img_path,'r',encoding='utf-8') as f_img,\
             open(box_path,'r',encoding='utf-8') as f_box:

            words,labels,boxes = [],[],[]
            curr_file_name = None
            img = None

            for (line_text,line_img,line_box) in zip(f_text,f_img,f_box):
                if line_text.strip() == "":
                    if words:
                        words_per_doc.append(words)
                        labels_per_doc.append(labels)
                        bboxes_per_doc.append(boxes)
                        images.append(img)
                        words,labels,boxes = [],[],[]
                    continue

                word,label = line_text.strip().split("\t")
                word_box = list(map(int, line_box.strip().split("\t")[1].split()))
                img_info = line_img.strip().split("\t")
                filename = img_info[-1]

                if curr_file_name != filename:
                    curr_file_name = filename
                    img = Image.open(os.path.join(img_dir,filename)).convert("RGB")
                
                words.append(word)
                labels.append(label)
                boxes.append(word_box)
        
        print("Dataset loaded successfully")
        return words_per_doc,labels_per_doc,bboxes_per_doc,images
                    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self,idx):
        encoding = self.processor(
            self.images[idx],
            self.words[idx],
            boxes=self.bboxes[idx],
            word_labels=[self.label2id[label] for label in self.labels[idx]],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        return {k:v.squeeze(0) for k,v in encoding.items()}