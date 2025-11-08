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
df = df.drop(df[~df['image'].isin(adjusted_files_in_use)].index)
df = df.head(10)

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


image_inputs = []
video_inputs = []
for message in messages:
    image_input, video_input = process_vision_info(message)
    image_inputs.append(image_input)
    video_inputs.append(video_input)

input = processor(
        text=text[0],
        images=image_inputs[0],
        videos=video_inputs[0],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    


generated_id = model.generate(**input, use_cache=True, max_new_tokens=1024, do_sample=False, generation_config=temp_generation_config)

generated_id_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input.input_ids, generated_id)]

print(len(generated_id_trimmed))
output_text = processor.batch_decode(generated_id_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(f'model output: {output_text}')
print(output_text.index("<answer>"))

open_tag = processor(
        text="<answer>",
        images=image_inputs[0],
        videos=video_inputs[0],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

close_tag = processor(
        text="</answer>",
        images=image_inputs[0],
        videos=video_inputs[0],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

random_jargon = processor(
        text="The man in the high castle",
        images=image_inputs[0],
        videos=video_inputs[0],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

print(open_tag['input_ids'])
print(close_tag['input_ids'])
print(random_jargon['input_ids'])