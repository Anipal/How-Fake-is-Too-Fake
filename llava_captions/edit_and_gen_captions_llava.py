import sys
import json
import tempfile
from accelerate.logging import get_logger
from accelerate import init_empty_weights
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import argparse
import difflib
import re
import random
import pdb
import torch
import os

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration 

# from lance.utils.misc_utils import *
torch.set_float32_matmul_precision("medium")
logger = get_logger("lance")

torch.set_default_tensor_type('torch.cuda.FloatTensor')

sys.path.append('/raid/mphute6/HalfTruths')
sys.path.append('/raid/mphute6/HalfTruthsLLaVA')
sys.path.append('/raid/mphute6/HalfTruths/LLaVA/llava/')

from llava.eval.run_llava import eval_model ,load_model, eval_model_wo_loading

# HERMES
LLAVA_PATH = "liuhaotian/llava-v1.6-34b"
CACHE_DIR = "cache"

context = """
Semantic change in images refers to alterations in meaning or interpretation. Here's how you might describe small, medium, and large semantic changes to an image:

1. Small Semantic Change: Could be something like changing the color of a specific object, adding or removing a minor detail, adjusting the composition or perspective of the image, or slightly adjusting the color distribution of the image. This change would not significantly alter the overall meaning or context of the image.
2. Medium Semantic Change: Could involve minor changes to object or setting characteristics, altering a background element, moving an object or person to another location within the frame, or changing the emotions of the people in the frame. This change could slightly alter the way the viewer perceives the image and its subject matter.
3. Large Semantic Change: This could include altering, adding or removing major elements of the image, such as people, objects, and landmarks, or making fundamental changes to the image's color scheme, lighting, or other visual elements. Involves substantial modifications that fundamentally transform the interpretation or message conveyed by the image. It can appear surprising or strange to an audience.

Using the definitions of small, medium, and large semantic changes above. """

# tokenizer_llama = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# model_llama = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


# pipeline = transformers.pipeline(
#             "text-generation",
#             tokenizer=tokenizer_llama,
#             model=model_llama,
#             torch_dtype=torch.float16,
#             device=0,
#         )

class LlavaGenerate_Edit:
    def __init__(
        self,
        args: argparse.Namespace,
        verbose: bool = False,
        prompt: str = "",
        image_file: str = "",
        cache_dir: str = "cache",
        model_name: str = "LlavaLlamaForCausalLM",
        model_path: str =  LLAVA_PATH,
        model_base = None,
        conv_mode = None,
        sep: str = ",",
        temp: int = 0,
        top_p = None,
        num_beams: int = 1,
        max_new_tokens: int = 512,
        sentence_embedding_model='sentence-transformers/all-MiniLM-L6-v2'
    ):
        self.prompt = prompt
        self.args = type('Args', (), {
            "model_path": model_path,
            "model_base": model_base,
            "model_name": model_name,
            "query": prompt,
            "conv_mode": conv_mode,
            "image_file": image_file,
            "sep": sep,
            "temperature": temp,
            "top_p": top_p,
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens ,
            "cache_dir" : cache_dir
        })()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_model(self.args)
        self.modified_caption_pattern = r'Modified Caption: (.+?)\n\n'
        self.original_caption_pattern = r'Original caption: (.+?)\n\n'
        self.sizes = ["small","medium","large"]
        self.semantic_change = SentenceTransformer(sentence_embedding_model)
        self.verbose = verbose

    def _compute_embeddings(self,original_cap, perturbed_caps):
        sentences=[]
        sentences.append(original_cap)
        sentences.extend(perturbed_caps)
        embeddings = self.semantic_change.encode(sentences)
        return embeddings

    '''
    def get_llama_response(self, prompt_eng, pipeline, tokenizer):
        instruction_prompt = prompt_eng
        response = pipeline(
            instruction_prompt,
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
            pad_token_id=tokenizer_llama.eos_token_id
        )   
        response_text = response[0]["generated_text"].split(":")[-1]
        # pdb.set_trace()
        return response_text
    '''

    def _compute_diff(self, cap1, cap2):
        """
        Computes the difference between two captions.
        """
        words_changed = list(
            difflib.ndiff(
                cap1.lower().replace(".", "").strip().split(" "),
                cap2.lower().replace(".", "").strip().split(" "),
            )
        )
        diff1, diff1_len, maxdiff1, maxdiff1_len = [], 0, [], 0
        diff2, diff2_len, maxdiff2, maxdiff2_len = [], 0, [], 0
        for word in words_changed:
            if "+" in word and len(word) > 2:
                diff1.append(word[2:])
                diff1_len += 1
                if diff1_len > maxdiff1_len:
                    maxdiff1_len = diff1_len
                    maxdiff1 = diff1
            else:
                diff1_len = 0
                diff1 = []

            if "-" in word and len(word) > 2:
                diff2.append(word[2:])
                diff2_len += 1
                if diff2_len > maxdiff2_len:
                    maxdiff2_len = diff2_len
                    maxdiff2 = diff2
            else:
                diff2_len = 0
                diff2 = []

        if len(maxdiff1):
            edit_word = " ".join(maxdiff1)
        else:
            edit_word = ""

        if len(maxdiff2):
            original_word = " ".join(maxdiff2)
        else:
            original_word = ""

        return original_word, edit_word

    def _parse_perturbations(self, original_cap, perturbed_caps, edit_types):
        """
        Parses the perturbations into a list of dictionaries.
        """
        # pdb.set_trace() 
        perturbations_dict_list = []
        sem_embed = self._compute_embeddings(original_cap, perturbed_caps)
        for ix, pcap in enumerate(perturbed_caps):
            original, edit = self._compute_diff(original_cap, pcap)
            sem_change = 1-(util.pytorch_cos_sim(sem_embed[0],sem_embed[ix+1])).cpu().numpy()[0][0]
            perturbations_dict_list.append(
                {
                    "original_caption": original_cap,
                    "original": original,
                    "edited_caption": pcap,
                    "edit": edit,
                    "sem_change":sem_change,
                    "perturbation_type": edit_types[ix],
                }
            )
        return perturbations_dict_list
    def generate_prompt_edits(self, og_cap, size):

        output_struct = f"""
            Return the captions in the format:
            {size} semantic change 1:
            Caption for changed image 1:

            {size} semantic change 2:
            Caption for changed image 2:

            {size} semantic change 3:
            Caption for changed image 3:

            """
        # prompt = context + f" Given the original caption for the image is: {og_cap}. Make 3 {size} semantic unique changes in the image, " + " The captions should be understandable in isolation to the original caption. Do not use relative descriptors" + output_struct
        prompt = context + f" Given the original caption for the image is: {og_cap}. Produce a new caption for this image that would reflect a {size} semantic change. Changes can be made to objects, verbs, adjectives, and prepositions" + output_struct
        return prompt
     
    def generate_prompt_og(self):
        prompt = f" Given the image, write a short caption for the image in under 20 words."
        return prompt
    
    def summarize_edited_caption(self, edited_cap):
        prompt = f"Rephrase the following text into one sentence based on the changes introduced in the text. Output only one sentence under 30 words: \n{edited_cap} "
        return prompt
    
    def get_blip_caption(self, img_file):
        raw_image = Image.open(img_file).convert('RGB')
        text = "a photo of"
        inputs = processor_blip(raw_image,text, return_tensors="pt").to("cuda")
        out = model_blip.generate(**inputs)
        caption_og = processor_blip.decode(out[0], skip_special_tokens=True)
        return caption_og
    
    def quality_flag(self, new_caps_list, perturbation_type):
        if perturbation_type == "all":
            perturbation_type_list = ["small", "medium", "large"]
        else:
            perturbation_type_list = perturbation_type
        
        ## making sure all types of requested perturbations are present
        ecap_ptypes = [ecap["perturbation_type"] for ecap in new_caps_list]
        
        if_type_in_ecap = [True if typ in ecap_ptypes else False for typ in perturbation_type_list]
        qual_flag = sum(if_type_in_ecap) == len(if_type_in_ecap)
        # pdb.set_trace()
        new_caps_list.append({"quality_flag": qual_flag})
        return new_caps_list
    
    def fix_caption_format(self, perturbed_captions_dict_list):
        for ix in range(len(perturbed_captions_dict_list)):
            perturbed_captions_dict_list[ix]["edited_caption"] = perturbed_captions_dict_list[ix]["edited_caption"].strip().split("\n")[0].split(".")[0]
        return perturbed_captions_dict_list

        
    def get_captions(self, image_file,perturbation_type='all'):
        perturbations = []
        edit_type = []
        # prompt_og = self.generate_prompt_og()
        # setattr(self.args,"image_file", image_file) 
        # setattr(self.args,"query",prompt_og) 
        # original_caption_whole = eval_model_wo_loading(self.tokenizer, self.model, self.image_processor, self.context_len, self.args)
        # original_caption = original_caption_whole.split(".")[0]
        # pdb.set_trace()
        original_caption = self.get_blip_caption(image_file)
        # pdb.set_trace()

        if perturbation_type == "all":
            perturbation_types = self.sizes
        else:
            perturbation_types = [perturbation_type]
        # print(f"INFO:  => Generated caption: {original_caption}")
        for perturbation_type in perturbation_types:
            modified_captions = []
            modified_summarized_captions = []
        
            prompt_edit = self.generate_prompt_edits(original_caption,perturbation_type)
            setattr(self.args,"query",prompt_edit) 
            setattr(self.args,"image_file", image_file) 
            # setattr(self.args,"image_file", "")
            responses = eval_model_wo_loading(self.tokenizer, self.model, self.image_processor, self.context_len, self.args)
            lines = responses.split('\n')
            if len(lines) == 0:
                print("WARNING: No edited captions found")
                # pdb.set_trace()
            else:
                for j, line in enumerate(lines):
                    line = line.strip()
                    if line.startswith('Caption for changed image'):
                        if(line.split(':')[-1] == ""):
                            try:
                                modified_captions.append(lines[j+1].strip())
                            except:
                                # pdb.set_trace()
                                print(f"skipped {perturbation_type}")
                                pass
                        else:
                            modified_captions.append(line.split(':')[-1])
                    
                        # summarize the modified caption and us ethat for compute instead
                        # print("unsummarized caption: ", modified_captions[-1])
                        # sum_edcap_prompt = self.summarize_edited_caption(modified_captions[-1])
                        # setattr(self.args,"query",sum_edcap_prompt) 
                        # sum_edcap = self.get_llama_response(sum_edcap_prompt, pipeline, tokenizer_llama) 
                        # pdb.set_trace()
                        modified_summarized_captions.append(modified_captions[-1])
            perturbations.extend(modified_summarized_captions)
            edit_type.extend([perturbation_type]*len(modified_summarized_captions))
                        
            # perturbations.extend(modified_captions)
            # edit_type.extend([perturbation_type]*len(modified_captions))
            
        if (len(perturbations) == 0):
            perturbed_captions_dict_list = {"empty":"empty"}
            return "", perturbed_captions_dict_list
        perturbed_captions_dict_list = self._parse_perturbations(original_caption , perturbations,edit_type)
        random.shuffle(perturbed_captions_dict_list)
        #fix formatting of the perturbed captions
        perturbed_captions_dict_list = self.fix_caption_format(perturbed_captions_dict_list)
        if self.verbose:
            for ix in range(min(len(perturbed_captions_dict_list), 10)):
                ecap = perturbed_captions_dict_list[ix]["edited_caption"]
                print(f"INFO ({edit_type[ix]}) :  {ix+1}: {ecap}")
            print(f"INFO: (end of image)...\n")
        if self.verbose:
            print(f"INFO:  => Original caption: {original_caption}")
            print("--------------------------------------------------------------")
        return original_caption, perturbed_captions_dict_list


        

