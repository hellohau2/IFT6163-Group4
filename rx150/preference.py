import io
import torch
import openai
import base64
import numpy as np
from PIL import Image
from qwen_vl_utils import process_vision_info
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from google import genai
from torchvision.transforms.functional import to_pil_image

# def get_first_prompt(goal_desc):
#     return f'''
#     1. What is shown in Image 1?
#     2. What is shown in Image 2?
#     3. The goal is {goal_desc}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
#     '''

# def get_last_prompt(goal_desc,first_prompt_answer):
#     return f"""
#     Based on the text below to the questions:
#     1. What is shown in Image 1?
#     2. What is shown in Image 2?
#     3. The goal is {goal_desc}. Is there any difference between Image 1 and Image 2 in terms of achieving the goal?
#     {first_prompt_answer}

#     Is the goal better achieved in Image 1 or Image 2?
#     Reply with only 0 if the goal is better achieved in Image 1, or 1 if it is better achieved in Image 2.
#     Reply -1 if you are unsure or there is no difference between the images.
#     """ 


def get_first_prompt(goal_desc):
    return f"""
The primary goal is: **{goal_desc}**

Analyze Image 1 and Image 2 with this goal in mind.

1.  **Image 1 Analysis:** Describe the state of the objects (e.g., blocks, robot gripper) and their positions *specifically in relation to achieving the goal*.
2.  **Image 2 Analysis:** Describe the state of the objects (e.g., blocks, robot gripper) and their positions *specifically in relation to achieving the goal*.
3.  **Comparison for Goal Achievement:** Directly compare Image 1 and Image 2. Which image represents a state closer to achieving the goal? Explain your reasoning, focusing strictly on the factors relevant to the goal. If both images are equally close/far from the goal, or if the difference is negligible for achieving the goal, state that clearly.
    """

def get_last_prompt(goal_desc, first_prompt_answer):
    return f"""
You previously analyzed the images with the goal "{goal_desc}" and provided this analysis:
---
{first_prompt_answer}
---

Now, based *only* on which image better achieves the goal state **{goal_desc}**, provide your final decision:

- Choose '0' if Image 1 is significantly closer to achieving the goal.
- Choose '1' if Image 2 is significantly closer to achieving the goal.
- Choose '-1' if both images are approximately equal in relation to the goal (e.g., equally far, equally close, or the difference is irrelevant to the goal).

Reply with only the single number (0, 1, or -1) and nothing else.
    """

class QwenPref():
    def __init__(self,max_tokens=512, goal_desc=""):
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            # attn_implementation="flash_attention_2",
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="cuda",
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
        self.max_tokens=max_tokens
        self.goal_desc=goal_desc

    def get_answer(self,pil_img_1,pil_img_2,prompt):
        messages=[
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Consider the following two images: Image 1:",
                },

                {
                    "type": "image",
                    "image": pil_img_1
                },

                {
                    "type": "text",
                    "text": "Image 2:",
                },

                {
                    "type": "image",
                    "image": pil_img_2
                },

                {
                    "type": "text",
                    "text": prompt,
                },
            ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=False,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

    def get_preference(self,img_1,img_2):
        # Convert to pillow Images
        pil_img_1 = Image.fromarray(img_1)
        pil_img_2 = Image.fromarray(img_2)

        prompt_1 = get_first_prompt(self.goal_desc)
        first_answer = self.get_answer(pil_img_1,pil_img_2,prompt_1)

        # print("Prompt 1 response:", first_answer)

        prompt_2 = get_last_prompt(self.goal_desc, first_answer)
        second_answer = self.get_answer(pil_img_1,pil_img_2,prompt_2)

        # print("Prompt 2 response:", second_answer)
        return int(second_answer)

class GPTPref():
    def __init__(self,api_key,model_name,temperature=0,detail='low',max_tokens=256, goal_desc=""):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.detail = detail
        self.max_tokens = max_tokens
        self.goal_desc = goal_desc
    
    def encode_image(self,tensor):    
        np_array = tensor.permute(1, 2, 0).detach().cpu().numpy()
        img = Image.fromarray(np_array.astype(np.uint8)).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def gpt_api_call(self,base64_image1, base64_image2, prompt):

        response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Consider the following two images: Image 1:",
                        },

                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image1}",
                                "detail": self.detail
                            },
                        },

                        {
                            "type": "text",
                            "text": "Image 2:",
                        },

                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image2}",
                                "detail": self.detail
                            },
                        },

                        {
                            "type": "text",
                            "text": prompt,
                        },
                        
                    ],
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        
        return response.choices[0].message.content

    def get_preference(self,img_1,img_2):

        b64_img_1 = self.encode_image(img_1)
        b64_img_2 = self.encode_image(img_2)

        prompt_1 = get_first_prompt(self.goal_desc)
        first_answer = self.gpt_api_call(b64_img_1, b64_img_2, prompt_1)

        # print("Prompt 1 response:", first_answer)

        prompt_2 = get_last_prompt(self.goal_desc, first_answer)
        second_answer = self.gpt_api_call(b64_img_1, b64_img_2, prompt_2)

        # print("Prompt 2 response:", second_answer)
        return int(second_answer)

class GeminiPref():

    def __init__(self,api_key,model_name, goal_desc=""):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.goal_desc = goal_desc
            
    def get_answer(self,pil_img_1,pil_img_2,prompt):

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=["Consider the following two images: Image 1:",
                pil_img_1,
                "Image 2: ",
                pil_img_2,
                prompt],
            # temperature=temperature,
            # max_tokens=max_tokens,
            )
        
        return response.text

    def get_preference(self,rx_img_1,rx_img_2):

        # pil_img_1 = Image.fromarray(rx_img_1)
        # pil_img_2 = Image.fromarray(rx_img_2)

        # If inputs are tensors
        pil_img_1 = to_pil_image(rx_img_1)
        pil_img_2 = to_pil_image(rx_img_2)

        prompt_1 = get_first_prompt(self.goal_desc)
        first_answer = self.get_answer(pil_img_1,pil_img_2,prompt_1)

        # print("Prompt 1 response:", first_answer)

        prompt_2 = get_last_prompt(self.goal_desc, first_answer)
        second_answer = self.get_answer(pil_img_1,pil_img_2,prompt_2)

        # print("Prompt 2 response:", second_answer)
        # return int(second_answer)
        return 1 if '1' in second_answer else 0 if '0' in second_answer else -1