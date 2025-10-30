from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
from qwen_vl_utils import process_vision_info
import torch

MODEL_PATH = 'JZPeterPan/MedVLM-R1'

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)

temp_generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=False,  
    temperature=1, 
    num_return_sequences=1,
    pad_token_id=151643,
)

questions = [{"image": ['images/mdb146.png'], "problem": "What content appears in this image?\nA) Cardiac tissue\nB) Breast tissue\nC) Liver tissue\nD) Skin tissue", "solution": "B", "answer": "Breast tissue"}, {"image": ["images/person19_virus_50.jpeg"], "problem": "What content appears in this image?\nA) Lungs\nB) Bladder\nC) Brain\nD) Heart", "solution": "A", "answer": "Lungs"},{"image":["images/abd-normal023599.png"],"problem":"Is any abnormality evident in this image?\nA) No\nB) Yes.","solution":"A","answer":"No"}, {"image":["images/foot089224.png"],"problem":"Which imaging technique was utilized for acquiring this image?\nA) MRI\nB) Electroencephalogram (EEG)\nC) Ultrasound\nD) Angiography","solution":"A","answer":"MRI"}, {"image":["images/knee031316.png"],"problem":"What can be observed in this image?\nA) Chondral abnormality\nB) Bone density loss\nC) Synovial cyst formation\nD) Ligament tear","solution":"A","answer":"Chondral abnormality"}, {"image":["images/shoulder045906.png"],"problem":"What can be visually detected in this picture?\nA) Bone fracture\nB) Soft tissue fluid\nC) Blood clot\nD) Tendon tear","solution":"B","answer":"Soft tissue fluid"}, {"image":["images/brain003631.png"],"problem":"What attribute can be observed in this image?\nA) Focal flair hyperintensity\nB) Bone fracture\nC) Vascular malformation\nD) Ligament tear","solution":"A","answer":"Focal flair hyperintensity"}, {"image":["images/mrabd005680.png"],"problem":"What can be observed in this image?\nA) Pulmonary embolism\nB) Pancreatic abscess\nC) Intraperitoneal mass\nD) Cardiac tamponade","solution":"C","answer":"Intraperitoneal mass"}]

QUESTION_TEMPLATE = """
    {Question} 
    Your task: 
    1. Think through the question step by step, enclose your reasoning process in <think>...</think> tags. 
    2. Then provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.
    3. No extra information or text outside of these tags.
    """

messages = [[{
    "role": "user",
    "content": [{"type": "image", "image": f'file://{question["image"][0]}'}, {"type": "text","text": QUESTION_TEMPLATE.format(Question=question['problem'])}]
}] for question in questions]

text = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]

image_inputs = []
video_inputs = []
for message in messages:
    print(message)
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
for id in generated_ids_trimmed:
    output_text = processor.batch_decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_texts.append(output_text)
    print(f'model output: {output_text}')
