from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os

df = pd.read_json("hf://datasets/FreedomIntelligence/Medical_Multimodal_Evaluation_Data/medical_multimodel_evaluation_data.json")
files_in_use = os.listdir("../image_mri/test")
adjusted_files_in_use = [["images/" + file] for file in files_in_use]
print(df["image"][2])
df = df.drop(df[~df['image'].isin(adjusted_files_in_use)].index)
print(df.shape)
df = df.head(10)
print(df.shape)

MODEL_PATH = 'JZPeterPan/MedVLM-R1'

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

device = torch.device("cuda:0")

processor = AutoProcessor.from_pretrained(MODEL_PATH)

temp_generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=False,  
    temperature=1, 
    num_return_sequences=1,
    pad_token_id=151643,
)

prefix = ["A", "B", "C", "D", "E", "F", "G"]
def options_maker(options):
    output_text = ""
    for i in range(len(options)):
        output_text += "\n" + prefix[i] + ") " + options[i]
    return output_text

questions = [{"image": "../image_mri/test/" + row["image"][0].split("/")[1], "problem": row["question"] + options_maker(row["options"]), "solution": prefix[row["options"].index(row["answer"])], "answer": row["answer"]} for index, row in df.iterrows()]
QUESTION_TEMPLATE = """
    {Question} 
    Your task: 
    1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 
    2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
    3. No extra information or text outside of these tags.
    """

messages = [[{
    "role": "user",
    "content": [{"type": "image", "image": f"file://{question['image']}"}, {"type": "text","text": QUESTION_TEMPLATE.format(Question=question['problem'])}]
}] for question in questions]

text = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
print(text[0])

image_inputs = []
video_inputs = []
for message in messages:
    image_input, video_input = process_vision_info(message)
    image_inputs.append(image_input)
    video_inputs.append(video_input)

inputs = []
for i in range(len(image_inputs)):
    input = processor(
        text=text[i],
        images=image_inputs[i],
        videos=video_inputs[i],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    inputs.append(input)

generated_ids = []
for input in inputs:
    generated_id = model.generate(**input, use_cache=True, max_new_tokens=1024, do_sample=False, generation_config=temp_generation_config)
    generated_ids.append(generated_id)

generated_ids_trimmed = []
for i in range(len(inputs)):
    generated_id_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs[i].input_ids, generated_ids[i])]
    generated_ids_trimmed.append(generated_id_trimmed)

output_texts = []
count = 0
for i in range(len(generated_ids_trimmed)):
    output_text = processor.batch_decode(generated_ids_trimmed[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_texts.append(output_text)
    print(f'model output: {output_text}')
    try:
        answer = output_text[0][output_text[0].index("<answer>") + 8]
        print(answer)
        print(questions[i]["solution"])
        if questions[i]["solution"] == answer:
            count +=1
    except ValueError:
        continue
print(count/len(generated_ids_trimmed))

selected_images = ["../image_mri/test/" + filename[0].split("/")[1] for filename in list(df.image)]
    
imagenet_path = "../image_mri/"
image_size = 256
dataset = torchvision.datasets.ImageFolder(root=imagenet_path, transform=transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)]))
indices = [i for i, (imgs) in enumerate(dataset.imgs) if imgs[0] in selected_images]
filtered_subset = torch.utils.data.Subset(dataset, indices)
""" data_loader = torch.utils.data.DataLoader(filtered_subset, shuffle=False, drop_last=False)

exmp_batch, _ = next(iter(data_loader))

inp_imgs = exmp_batch.clone().requires_grad_() """
inp_imgs = [img for img, _ in filtered_subset]


target_text = np.full(1, "A cancerous tumour").tolist()
enc = processor(images=inp_imgs, text=target_text, padding=True, return_tensors="pt")

pixel_values = enc["pixel_values"].to(device)
image_grid_thw = enc['image_grid_thw'].to(device)
pixel_values = pixel_values.clone().detach().requires_grad_(True)
input_ids = enc["input_ids"].to(device)
attention_mask = enc["attention_mask"].to(device)

""" row = torch.tensor([1, 256, 256], dtype=torch.long)
image_grid_thw = row.unsqueeze(0).repeat(len(filtered_subset), 1) """

print(image_grid_thw.shape)
print(pixel_values.shape)


kwargs = {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw, "input_ids": input_ids, "attention_mask": attention_mask}
print(image_grid_thw.shape)

output = model(**kwargs)


loss=output.loss
model.zero_grad()
loss.backward()

grad_sign = pixel_values.grad.detach().sign()

adv_pixel_values = pixel_values + 0.02*grad_sign

adv_pixel_values = torch.clamp(adv_pixel_values, 0.0, 1.0)

adv_pixel_values = adv_pixel_values.to(device)

batch_size = adv_pixel_values.shape[0]

generated_texts = []
for b in range(batch_size):
    kwargs = {
        "pixel_values": adv_pixel_values[b:b+1],
        "input_ids": input_ids[b:b+1],
        "attention_mask": attention_mask[b:b+1],
        "generation_config": temp_generation_config,
        "use_cache": True,
    }
    gen_ids = model.generate(**kwargs)
    text_out = processor.batch_decode(gen_ids, skip_special_tokens=True)
    generated_texts.append(text_out[0])

print(generated_texts)