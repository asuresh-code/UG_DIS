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

frequency_image_tensors = []
grey_image_tensors = []
image_inputs = []
video_inputs = []
count = 0
lower_bound_budgets = []
upper_bound_budgets = []
total_budget = 5
iterations = 10
alpha = 2
for message in messages:
    count += 1
    image_input, video_input = process_vision_info(message)
    transform = transforms.Compose([transforms.PILToTensor()])
    image_tensor = transform(image_input[0])
    freq_image_tensor = torch.fft.fft2(image_tensor[0]).float().clone().detach().to(device).requires_grad_(True)
    print("FREQ 1:", freq_image_tensor.shape)
    inversed_tensor = torch.fft.ifft2(freq_image_tensor).real
    grey_image_tensor = inversed_tensor.unsqueeze(0)
    print("1: ", grey_image_tensor)
    image_tensor = grey_image_tensor.repeat(3,1,1)
    lower_bound_image_tensor = grey_image_tensor.clone().detach() - total_budget
    upper_bound_image_tensor = grey_image_tensor.clone().detach() + total_budget
    image_inputs.append(image_tensor)
    video_inputs.append(video_input)
    lower_bound_budgets.append(lower_bound_image_tensor)
    upper_bound_budgets.append(upper_bound_image_tensor)
    grey_image_tensors.append(grey_image_tensor)
    frequency_image_tensors.append(freq_image_tensor)

for i in range(iterations):
    inputs = []
    for x in range(len(image_inputs)):
        input = processor(
            text=text[x],
            images=image_inputs[x],
            videos=video_inputs[x],
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
    for x in range(len(generated_ids)):
        output_text = processor.batch_decode(generated_ids[x][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if i == 9:
            print(output_text)
        output_texts.append(output_text)

    for x in range(len(generated_ids)):
        sequence = []
        for token in generated_ids_grad[x]['logits'][0]:
            sequence.append(token.argmax())
        tokenizer = processor.tokenizer
        string_tokens = tokenizer.convert_ids_to_tokens(sequence)
        answer_token_pos = find_answer_token(string_tokens)
        if answer_token_pos == -1:
            print("No answer found")
            continue
        else:
            actual_answer = questions[x]['solution']
            if len(string_tokens[answer_token_pos]) > 1:
                actual_answer = ">" + actual_answer
            print(actual_answer)
            if actual_answer == string_tokens[answer_token_pos]:
                successes += 1
        logits_vec = generated_ids_grad[x]["logits"][0, answer_token_pos, :] 

        label_scalar = torch.tensor(tokenizer.convert_tokens_to_ids(actual_answer)).to(device)
        loss = torch.nn.functional.cross_entropy(
            logits_vec.unsqueeze(0),
            label_scalar.unsqueeze(0),
        )
        model.zero_grad()
        loss.backward()

        signed_grad_freq = torch.sign(frequency_image_tensors[x].grad)

        frequency_image_tensors[x].grad = None

        base_signed_pix = torch.fft.ifft2(signed_grad_freq).real

        factor = 2.0 / torch.max(base_signed_pix)
        print("factor:", factor)

        new_freq_tensor = frequency_image_tensors[x].clone().detach() + signed_grad_freq*factor
        print("2 FREQ TESNRO:", new_freq_tensor.shape)

        adv_image = torch.fft.ifft2(new_freq_tensor).real.unsqueeze(0)
        print("3:", adv_image.shape)
        print("lbb:", lower_bound_budgets[x].shape)

        lower_bound_pos = torch.lt(adv_image, lower_bound_budgets[x])
        non_lower_bound_pos = torch.logical_not(lower_bound_pos)

        component1 = torch.multiply(lower_bound_pos, lower_bound_budgets[x])
        component2 = torch.multiply(non_lower_bound_pos, adv_image)

        final_image = torch.add(component1, component2)

        upper_bound_pos = torch.gt(final_image, upper_bound_budgets[x])
        non_upper_bound_pos = torch.logical_not(upper_bound_pos)

        component1 = torch.multiply(upper_bound_pos, upper_bound_budgets[x])
        component2 = torch.multiply(non_upper_bound_pos, final_image)

        final_image = torch.add(component1, component2)

        final_image = torch.clamp(final_image, min=0, max=255)

        frequency_image_tensors[x] = torch.squeeze(torch.fft.fft2(final_image).float().clone()).detach().to(device).requires_grad_(True)
        print("3 FREQ IMAGE TENSRO:", frequency_image_tensors[x].shape)
        inversed_tensor = torch.fft.ifft2(frequency_image_tensors[x]).real
        print("4:", inversed_tensor.shape)
        grey_image_tensors[x] = inversed_tensor
        image_inputs[x] = grey_image_tensors[x].clone().repeat(3,1,1)

    print("Success Rate:",successes/len(generated_ids))

transform = transforms.ToPILImage()
img = transform(image_inputs[0].to(torch.uint8))
sv = img.save(questions[0]["filename"])