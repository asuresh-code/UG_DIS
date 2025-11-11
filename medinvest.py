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
count = 0
for message in messages:
    print("Message content: ", message[0]["content"][0]["image"])
    count += 1
    image_input, video_input = process_vision_info(message)
    print("Processed image: ", image_input)
    transform = transforms.Compose([transforms.PILToTensor()])
    image_tensor = transform(image_input[0])
    print("Image Tesnro first: ", image_tensor)
    image_tensor = image_tensor.float().clone().detach().requires_grad_(True)
    print("Image Tesnro second: ", image_tensor)
    np_array = image_tensor.clone().detach().cpu().numpy()
    np.save(f"image_tensor_{count}.txt", np_array.flatten(), fmt='%.6f')
    image_inputs.append(image_tensor)
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
generated_ids_grad = []
for input in inputs:
    generated_id = model.generate(**input, use_cache=True, do_sample=False, generation_config=temp_generation_config, return_dict_in_generate=True, output_logits=True)
    sequence = generated_id['sequences']
    sequence.detach()
    sequence = sequence.to(device)
    attention_mask = torch.ones_like(sequence)
    generated_id_grad = model(input_ids=sequence, pixel_values=input['pixel_values'], attention_mask=attention_mask, image_grid_thw=input['image_grid_thw'])
    generated_ids.append(generated_id)
    generated_ids_grad.append(generated_id_grad)

output_texts = []
successes = 0
for i in range(len(generated_ids)):
    output_text = processor.batch_decode(generated_ids[i][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_texts.append(output_text)

adv_images = []
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
        print(actual_answer)
        if actual_answer == string_tokens[answer_token_pos]:
            successes += 1
    logits_vec = generated_ids_grad[i]["logits"][0, answer_token_pos, :] 
 
    label_scalar = sequence[answer_token_pos]
    loss = torch.nn.functional.cross_entropy(
        logits_vec.unsqueeze(0),
        label_scalar.unsqueeze(0),
    )
    print("Logits shape: ",logits_vec.unsqueeze(0).shape)
    print("label_scalar shape: ",label_scalar.unsqueeze(0).shape)
    print("logits Value: ",logits_vec.unsqueeze(0))
    print("label_scalar Value: ",label_scalar.unsqueeze(0))
    
    print(loss)
    print("Image input: ", image_inputs[i])
    print("Image input grad before: ", image_inputs[i].grad)
    loss.backward()
    print("Image input grad after: ", image_inputs[i].grad)
    print("Image needs grad: ", image_inputs[i].requires_grad)
    print("Image is leaf tensor: ", image_inputs[i].is_leaf)
    print("Image has grad fn: ", image_inputs[i].grad_fn)
    signed_grad = torch.sign(image_inputs[i].grad)
    adv_image = image_inputs[i].clone().detach() + signed_grad
    adv_image = torch.clamp(adv_image, min=0, max=255)
    adv_images.append(adv_image)

print("Success Rate:",successes/len(generated_ids))

inputs = []
for i in range(len(image_inputs)):
    input = processor(
        text=text[i],
        images=adv_images[i],
        videos=video_inputs[i],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    inputs.append(input)

generated_ids = []
for input in inputs:
    generated_id = model.generate(**input, do_sample=False, generation_config=temp_generation_config, return_dict_in_generate=True, output_logits=True)
    generated_ids.append(generated_id)

successes = 0
for i in range(len(generated_ids)):
    output_text = processor.batch_decode(generated_ids[i][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    try:
        answer = output_text[0][output_text[0].rindex("<answer>") + 8]
        print(output_text[0])
        print(output_text[0][output_text[0].rindex("<answer>")])
        print(answer)
        print(questions[i]["solution"])
        if questions[i]["solution"] == answer:
            successes +=1
    except ValueError:
        continue
print("Success Rate After Adv:",successes/len(generated_ids))