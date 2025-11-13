from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
from PIL import Image

df = pd.read_json("hf://datasets/FreedomIntelligence/Medical_Multimodal_Evaluation_Data/medical_multimodel_evaluation_data.json")
files_in_use = os.listdir(f"../image_mri/test")
adjusted_files_in_use = [["images/" + file] for file in files_in_use]
df = df.drop(df[~df['image'].isin(adjusted_files_in_use)].index)
df = df.head(10)

start_tag = [27,9217,29]
end_tag = [522,9217,29]

torch.set_grad_enabled(True)
torch.autograd.set_detect_anomaly(True)

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
    torch_dtype=torch.float32,
    device_map="cuda:0",
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
                """ print(token_list[i]) """
                token_target_pos = i
                break
        else:
            if "answer" in token_list[i]:
                if count > 0:
                    count -= 1
                else:
                    found_start = True
    return token_target_pos


questions = [{"filename": row["image"][0].split("/")[1],"image": "../image_mri/test/" + row["image"][0].split("/")[1], "problem": row["question"] + options_maker(row["options"]), "solution": prefix[row["options"].index(row["answer"])], "answer": row["answer"]} for index, row in df.iterrows()]
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
    transform = transforms.Compose([transforms.PILToTensor()])
    image_tensor = transform(image_input[0])
    image_tensor = image_tensor.float().clone().detach().to(device).requires_grad_(True)
    image_inputs.append(image_tensor)
    video_inputs.append(video_input)

for i in range(10):
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
    generated_ids_grad = []
    for input in inputs:
        generated_id = model.generate(**input, use_cache=True, do_sample=False, generation_config=temp_generation_config, return_dict_in_generate=True, output_logits=True)
        sequence = generated_id['sequences']
        sequence.clone().detach()
        sequence = sequence.to(device)
        attention_mask = torch.ones_like(sequence)
        generated_id_grad = model(input_ids=sequence, pixel_values=input['pixel_values'], attention_mask=attention_mask, image_grid_thw=input['image_grid_thw'])
        generated_ids.append(generated_id)
        generated_ids_grad.append(generated_id_grad)

    output_texts = []
    successes = 0
    for i in range(len(generated_ids)):
        output_text = processor.batch_decode(generated_ids[i][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if i == 0:
            print(output_text)
            print(questions[i]['filename'])
        output_texts.append(output_text)

    for i in range(len(generated_ids)):
        sequence = []
        for token in generated_ids_grad[i]['logits'][0]:
            sequence.append(token.argmax())
        tokenizer = processor.tokenizer
        string_tokens = tokenizer.convert_ids_to_tokens(sequence)
        answer_token_pos = find_answer_token(string_tokens)
        if answer_token_pos == -1:
            print("No answer found")
            continue
        else:
            actual_answer = questions[i]['solution']
            if len(string_tokens[answer_token_pos]) > 1:
                actual_answer = ">" + actual_answer
            """ print(actual_answer) """
            if actual_answer == string_tokens[answer_token_pos]:
                successes += 1
        logits_vec = generated_ids_grad[i]["logits"][0, answer_token_pos, :] 

        label_scalar = sequence[answer_token_pos]
        loss = torch.nn.functional.cross_entropy(
            logits_vec.unsqueeze(0),
            label_scalar.unsqueeze(0),
        )    
        loss.backward()
        signed_grad = torch.sign(image_inputs[i].grad)
        adv_image = image_inputs[i].clone().detach() + signed_grad
        adv_image = torch.clamp(adv_image, min=0, max=255)
        image_inputs[i] = adv_image


    print("Success Rate:",successes/len(generated_ids))