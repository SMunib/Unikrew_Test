from transformers import LayoutLMv2ForTokenClassification
import torch

class LayoutLM_Model:
    def __init__(self, model_path,num_labels,id2label,label2id,processor,dataset):
        print("Initializing LayoutLM_Model....")
        self.model_path = model_path
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.processor = processor
        self.dataset = dataset
        self.device =  "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self._load_model()

    def _load_model(self):
        
        self.model = LayoutLMv2ForTokenClassification.from_pretrained(
            self.model_path,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        ).to(self.device)
        self.model.eval()
        print("Model loaded successfully")
    
    def model_infer(self,sample_idx):
       
        if self.model is None:
            raise ValueError("Model is not loaded")
        # Get the selected sample
        sample = self.dataset[sample_idx]
        # Prepare input (exclude labels)
        inputs = {k: v.unsqueeze(0).to(self.evice) for k, v in sample.items() if k != "labels"}
        # Run inference
        with torch.inference_mode():
            outputs = self.model(**inputs)
            preds = outputs.logits.argmax(-1).squeeze().cpu().numpy()

        # Recreate encoding for alignment
        encoding = self.processor(
            self.dataset.images[sample_idx],
            self.dataset.words[sample_idx],
            boxes=self.dataset.bboxes[sample_idx],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        # Initialize expected fields
        results = {"company": "", "address": "", "total": "", "date": ""}

        # Track seen words to avoid duplicates
        seen = {k: set() for k in results.keys()}

        # Decode predictions
        for idx, label_id in enumerate(preds):
            word_idx = encoding.word_ids(batch_index=0)[idx]
            if word_idx is None:
                continue  # Skip special/padding tokens

            label = self.id2label[label_id]
            if label != "O":
                key = label.replace("S-", "").lower()
                if key in results:
                    word = self.dataset.words[sample_idx][word_idx]
                    if word not in seen[key]:
                        results[key] += " " + word
                        seen[key].add(word)

        # Clean whitespace
        results = {k: v.strip() for k, v in results.items()}
        return results
