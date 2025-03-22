 import av
import os
import numpy as np
import torch
from transformers import LlavaForConditionalGeneration, LlavaNextProcessor
import json
import pandas as pd
from PIL import Image

def read_images(image_paths):
    # Read images from the paths provided in the JSON
    images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
    return np.stack([np.array(img) for img in images])

def inference(model, processor, system_prompt, image_paths):
    # Define the conversation using the system prompt and images
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
            ] + [{"type": "image"} for _ in image_paths],
        },
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    for i, img_path in enumerate(image_paths):
        print(img_path)
        image_paths[i] = './data/image_jpg/' + img_path

    print(image_paths)
    # Read images using the helper function
    clip = read_images(image_paths)
    inputs_images = processor(text=prompt, images=clip, padding=True, return_tensors="pt").to(model.device)

    # Convert only 'pixel_values' to float16
    if 'pixel_values' in inputs_images:
        inputs_images['pixel_values'] = inputs_images['pixel_values'].half()

    # Generate the output from the model
    print(inputs_images.keys())
    output = model.generate(**inputs_images, max_new_tokens=512)
    generated_text = processor.decode(output[0], skip_special_tokens=True)

    # Filter to only include the text after "assistant"
    assistant_output = generated_text.split("assistant", 1)[-1].strip()

    return assistant_output



    
def process_json_and_generate_reports(json_path, model, processor, output_excel_path):
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = []

    # Iterate over each entry in the JSON
    for entry in data:
        system_prompt = entry['system_prompt']
        image_paths = entry['image']
        conversations = entry['conversations']
        
        # Extract the patient and study information
        patient = image_paths[0].split('_')[0]
        study = image_paths[0].split('_')[1]

        # Get the ground truth (GT) text from the JSON
        gt_text = next(conv['value'] for conv in conversations if conv['from'] == 'gpt')

        
        # Generate the model report
        generated_report = inference(model, processor, system_prompt, image_paths)

        # Store the information
        results.append({
            'patient': patient,
            'study': study,
            'gt': gt_text,
            'generated_report': generated_report
        })
    
    # Create a DataFrame and save it to an Excel file
    df = pd.DataFrame(results)
    df.to_excel(output_excel_path, index=False)
    print(f"Results saved to {output_excel_path}")

# Initialize the processor
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-interleave-qwen-7b-hf")

# Load the fine-tuned model
model = LlavaForConditionalGeneration.from_pretrained(
    "./checkpoints/llava-interleave-qwen-7b_0928final_lora-True_qlora-False",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(0)

# Path to the JSON file
json_path = '/raid/jupyter-charlielibear.md09-24f36/lmms-finetune/data/0921_externaleval_2.json'
# Path to save the output Excel file
output_excel_path = '2_model_report_comparison.xlsx'

# Process the JSON and generate reports
process_json_and_generate_reports(json_path, model, processor, output_excel_path)

# Clean up
del model