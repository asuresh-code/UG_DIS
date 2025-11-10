from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


df = pd.read_json("hf://datasets/FreedomIntelligence/Medical_Multimodal_Evaluation_Data/medical_multimodel_evaluation_data.json")
files_in_use = os.listdir("../image_mri/test")
adjusted_files_in_use = [["images/" + file] for file in files_in_use]
df = df.drop(df[~df['image'].isin(adjusted_files_in_use)].index)
df = df.head(10)

start_tag = [27,9217,29]
end_tag = [522,9217,29]

def splice_tokens(token_list):
    return_list = []
    start_found = [False, False, False]
    end_found = [False, False, False]
    for token in token_list:
        if start_found[0]:
            if start_found[1]:
                if start_found[2]:
                    return_list.append(token)
                    if end_found[0]:
                        if end_found[1]:
                            if end_found[2]:
                                continue
                            else:
                                if token == end_tag[2]:
                                    end_found[2] = True
                                else:
                                    end_found[0] = False
                                    end_found[1] = False
                        else:
                            if token == end_tag[1]:
                                end_found[1] = True
                            else:
                                end_found[0] = False
                    else:
                        if token == end_tag[0]:
                            end_found[0] = True
                else:
                    if token == start_tag[2]:
                        start_found[2] = True
                    else:
                        start_found[0] = False
                        start_found[1] = False
            else:
                if token == start_tag[1]:
                    start_found[1] = True
                else:
                    start_found[0] = False
        else:
            if token == start_tag[0]:
                start_found[0] = True
        if end_found[2]:
            break
    return return_list[:-3]

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

def find_answer_token(token_list):
    count = -2
    for token in token_list:
        if "answer" in token:
            count += 1

    token_target_pos = -1
    found_start = False
    for i in range(len(token_list)):
        if found_start:
            if token_list[i] != ">":
                if len(token_list[i]) > 1 and ord(token_list[i][1]) == 266:
                    continue
                print(token_list[i])
                token_target_pos = i
                break
        else:
            if "answer" in token_list[i]:
                if count > 0:
                    count -= 1
                else:
                    found_start = True
    return token_target_pos


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

inputs = []
for i in range(len(image_inputs)):
    input = processor(
        text=text[i],
        images=image_inputs[i],
        videos=video_inputs[i],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    input['pixel_values'] = input['pixel_values'].clone().detach().requires_grad_(True)
    inputs.append(input)

generated_ids = []
generated_ids_grad = []
for input in inputs:
    generated_id = model.generate(**input, use_cache=True, do_sample=False, generation_config=temp_generation_config, return_dict_in_generate=True, output_logits=True)
    sequence = generated_id['sequences']
    sequence.detach()
    sequence.to(device)
    generated_id_grad = model(input_ids=sequence)
    generated_ids.append(generated_id)
    generated_ids_grad.append(generated_id_grad)

output_texts = []
for i in range(len(generated_ids)):
    output_text = processor.batch_decode(generated_ids[i][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_texts.append(output_text)

for i in range(len(generated_ids)):
    print(len(generated_ids_grad[i]['logits'][0]))
    sequence = []
    for token in generated_ids_grad[i]['logits'][0]:
        sequence.append(token.argmax())
    tokenizer = processor.tokenizer
    string_tokens = tokenizer.convert_ids_to_tokens(sequence)
    print(string_tokens)
    answer_token_pos = find_answer_token(string_tokens)
    if answer_token_pos == -1:
        print("No answer found")
        continue
    logits_vec = generated_ids_grad[i]["logits"][0, answer_token_pos, :] 
 
    label_scalar = sequence[answer_token_pos]
    loss = torch.nn.functional.cross_entropy(
        logits_vec.unsqueeze(0),
        label_scalar.unsqueeze(0),
    )
    print(loss)
    loss.backward()
    signed_grad = torch.sign(inputs[i]['pixel_values'].grad)
    print(signed_grad)
    plt.imshow(signed_grad[0] * 0.5 + 0.5)
""" output_text = processor.batch_decode(answer_tensor, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(f'model output: {output_text}') """