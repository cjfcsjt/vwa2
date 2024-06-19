from typing import List

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)
import torch
import sys
sys.path.append('/mnt/nvme0n1p1/hongxin_li/agent-eval/')
from lmms_eval.models.model_utils.qwen.qwen_generate_utils import make_context
from transformers import AutoModelForCausalLM, AutoTokenizer
import uuid
import os

def get_captioning_fn(
    device, dtype, model_name: str = "Salesforce/blip2-flan-t5-xl"
) -> callable:
    if "blip2" in model_name:
        captioning_processor = Blip2Processor.from_pretrained(model_name)
        captioning_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        )
        captioning_model.to(device)
        def caption_images(
            images: List[Image.Image],
            prompt: List[str] = None,
            max_new_tokens: int = 32,
        ) -> List[str]:
            if prompt is None:
                # Perform VQA
                inputs = captioning_processor(
                    images=images, return_tensors="pt"
                ).to(device, dtype)
                generated_ids = captioning_model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
                captions = captioning_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
            else:
                # Regular captioning. Prompt is a list of strings, one for each image
                assert len(images) == len(
                    prompt
                ), "Number of images and prompts must match, got {} and {}".format(
                    len(images), len(prompt)
                )
                inputs = captioning_processor(
                    images=images, text=prompt, return_tensors="pt"
                ).to(device, dtype)
                generated_ids = captioning_model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
                captions = captioning_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

            return captions

    elif 'funcpred' in model_name:
        captioning_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True).eval() # load_in_4bit=True
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
        captioning_model.to(device)
        
        def caption_images(
            images,
            prompt,
            max_new_tokens: int = 32,
        ) -> List[str]:
            query = []
            visual_paths = []
            for visual in images:
                name = uuid.uuid4().hex.upper()[0:6]
                visual.save(f"/tmp/{name}.png")
                visual_paths.append(f"/tmp/{name}.png")
            prompt = f"In this web page image, please locate the element based on the \"{prompt}\" (with point)."
            query.append({"image": visual_paths[0]})
            query.append({"text": prompt})
            questions = tokenizer.from_list_format(query)
            # https://huggingface.co/cckevinn/SeeClick/blob/main/generation_config.json
            gen_kwargs = {}
            gen_kwargs["max_new_tokens"] = max_new_tokens
            gen_kwargs["temperature"] = 0.5
            gen_kwargs["top_p"] = None
            gen_kwargs["num_beams"] = 1
            gen_kwargs["chat_format"] = "chatml"
            gen_kwargs["do_sample"] = True
            gen_kwargs["eos_token_id"] = 151643
            gen_kwargs["max_window_size"] = 1024
            gen_kwargs["pad_token_id"] = 151643
            gen_kwargs["top_k"] = 0
            gen_kwargs["transformers_version"] = "4.36.2"
            text_output, history = captioning_model.chat(tokenizer, 
                                                        query=questions, 
                                                        history=None,
                                                        **gen_kwargs)
            for visual_path in visual_paths:
                try:
                    os.remove(visual_path)
                except:
                    pass
            return text_output
        
        def logits(
            images,
            prompt,
            multiple_choice,
            max_new_tokens: int = 32,
        ) -> List[str]:
            query = []
            visual_paths = []
            for visual in images:
                name = uuid.uuid4().hex.upper()[0:6]
                visual.save(f"/tmp/{name}.png")
                visual_paths.append(f"/tmp/{name}.png")
            prompt = f"In this web page image, please locate the element based on the \"{prompt}\" (with point)."
            query.append({"image": visual_paths[0]})
            query.append({"text": prompt})
            # split the choice into a list of strings
            choice_prob = []
            for choice in multiple_choice:
                # split the choice into a list of strings
                # for i in range(len(continuations)):
                #     if continuations[i] == ',':
                #         continue
                    # cur_cont = ''.join(continuations[:i])
                context_query = [
                    {"image": visual_paths[0]},
                    {"text": prompt}
                ]
                query = [
                    {"image": visual_paths[0]},
                    {"text": prompt + " " + choice}
                ]
                context_query = tokenizer.from_list_format(context_query)
                query = tokenizer.from_list_format(query)
                raw_contxt_text, context_tokens = make_context(
                tokenizer, context_query, history=None, system="You are a helpful assistant", max_window_size=captioning_model.generation_config.max_window_size, chat_format=captioning_model.generation_config.chat_format
                )
                context_tokens = torch.tensor([context_tokens])

                raw_continuation_text, continuation_tokens = make_context(
                    tokenizer, query, history=None, system="You are a helpful assistant", max_window_size=captioning_model.generation_config.max_window_size, chat_format=captioning_model.generation_config.chat_format
                )
                
                attn_mask = torch.ones_like(continuation_tokens).to(captioning_model.device)
                
                labels = continuation_tokens.clone().to(captioning_model.device)
                labels[:, : context_tokens.shape[1]] = -100 # leaves answer
                
                with torch.inference_mode():
                    outputs = captioning_model(input_ids=continuation_tokens, labels=labels, attention_mask=attn_mask)
                loss = outputs.loss
                logits = outputs["logits"]
                # greedy_tokens = logits.argmax(dim=-1)
                answer_logits = logits[:, context_tokens.shape[1] : continuation_tokens.shape[1]]  # [1, seq]
                for i in range(len(choice)):
                    if choice[i] == ',' or ')' or ')':
                        continue
                    index = tokenizer(choice[i])['input_ids'][0]
                    logit = logits[0, index]
                    choice_prob.append(logit)
            
            for visual_path in visual_paths:
                try:
                    os.remove(visual_path)
                except:
                    pass
            probs = 1
            for i in range(len(choice_prob)):
                probs = probs * choice_prob[i]
            return probs

    else:
        raise NotImplementedError(
            "Only BLIP-2 models are currently supported"
        )

    return caption_images


def get_image_ssim(imageA, imageB):
    # Determine the size to which we should resize
    new_size = max(imageA.size[0], imageB.size[0]), max(
        imageA.size[1], imageB.size[1]
    )

    # Resize images
    imageA = imageA.resize(new_size, Image.LANCZOS)
    imageB = imageB.resize(new_size, Image.LANCZOS)

    # Convert images to grayscale
    grayA = imageA.convert("L")
    grayB = imageB.convert("L")

    # Convert grayscale images to numpy arrays for SSIM computation
    grayA = np.array(grayA)
    grayB = np.array(grayB)

    # Compute the Structural Similarity Index (SSIM) between the two images
    score, _ = ssim(grayA, grayB, full=True)
    return score
