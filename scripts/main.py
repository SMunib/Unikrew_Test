from transformers import LayoutLMv2Processor
from modules.Dataset import ImageDataset
from modules.LayoutLM import LayoutLM_Model

label2id = {
    "O": 0,
    "S-COMPANY": 1,
    "S-DATE": 2,
    "S-ADDRESS": 3,
    "S-TOTAL": 4,
}
id2label = {v:k for k,v in label2id.items()}

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased",apply_ocr=False)

# train_dataset = ImageDataset(
#     txt_path="processed_data/train/train.txt",
#     img_path="processed_data/train/train_img.txt",
#     box_path="processed_data/train/train_box.txt",
#     processor=processor,
#     label2id=label2id,
#     img_dir="dataset/train/img"
# )
test_dataset = ImageDataset(
    txt_path="processed_data/test/test.txt",
    img_path="processed_data/test/test_img.txt",
    box_path="processed_data/test/test_box.txt",
    processor=processor,
    label2id=label2id,
    img_dir="dataset/test/img"
)

inference_model = LayoutLM_Model(
    model_path="modules/model",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    processor=processor,
    dataset=test_dataset
)

results = inference_model.model_infer(0)
print(results)