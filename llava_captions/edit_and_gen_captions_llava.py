import sys
import json
import tempfile
from accelerate.logging import get_logger
from sentence_transformers import SentenceTransformer, util
import argparse
import difflib
import re
import random
import pdb
from lance.utils.misc_utils import *
torch.set_float32_matmul_precision("medium")
logger = get_logger("lance")

sys.path.append('/srv/hoffman-lab/share4/apal72/half-truths')
sys.path.append('/srv/hoffman-lab/share4/apal72/half-truths/LLaVA')
sys.path.append('/srv/hoffman-lab/share4/apal72/half-truths/LLaVA/llava/')

from llava.eval.run_llava import eval_model ,load_model, eval_model_wo_loading

LLAVA_PATH = "liuhaotian/llava-v1.6-mistral-7b"
CACHE_DIR = "cache"


context = """
Semantic change in images refers to alterations in meaning or interpretation. Here's how you might describe small, medium, and large semantic changes to an image:

1. Small Semantic Change: Could be something like changing the color of a specific object, or adding or removing a minor detail, adjusting the composition or perspective of the image, or changing the distance or angle with which the image is taken, or slightly adjusting the color distribution of the image. This change would not significantly alter the overall meaning or context of the image.
2. Medium Semantic Change: Could involve moving an object or person to a different location within the frame, or changing the emotions of the people in the frame. Involves more noticeable alterations that can shift the focus or narrative of the image.
3. Large Semantic Change: This could include adding or removing major elements of the image, such as people, objects, or landmarks. Involves substantial modifications that fundamentally transform the interpretation or message conveyed by the image. It may even appear surprising or strange to an audience. 

Using the definitions of small, medium , and large semantic changes above,"""

# context = """
# Semantic change in images refers to alterations in meaning or interpretation. Here's how you might describe small, medium, and large semantic changes to an image:

# 1. Small Semantic Change: Could be something like changing the color of a specific object, or adding or removing a minor detail, adjusting the composition or perspective of the image, or changing the distance or angle with which the image is taken, or slightly adjusting the color distribution of the image. This change would not significantly alter the overall meaning or context of the image.
# 2. Medium Semantic Change: Could involve moving an object or person to a different location within the frame, or changing the emotions of the people in the frame. Involves more noticeable alterations that can shift the focus or narrative of the image.
# 3. Large Semantic Change: This could include adding or removing major elements of the image, such as people, objects, or landmarks, or making fundamental changes to the image's color scheme, lighting, or other visual elements. Involves substantial modifications that fundamentally transform the interpretation or message conveyed by the image. It may even appear surprising or strange to an audience.

# Using the definitions of small, medium, and large semantic changes above,"""

# output_struct = """

# Return the captions in the format:
# Original Caption:

# Modified Caption 1:
# Change 1:

# Modified Caption 2:
# Change 2:
# """

# context = """
# Semantic change in images refers to alterations in meaning or interpretation. Here's how you might describe small, medium, and large semantic changes to an image:

# 1. Small Semantic Change: Could be something like changing the color of a specific object, or adding or removing a minor detail, adjusting the composition or perspective of the image, or changing the distance or angle with which the image is taken, or slightly adjusting the color distribution of the image. This change would not significantly alter the overall meaning or context of the image.
# 2. Large Semantic Change: This could include adding or removing major elements of the image, such as people, objects, or landmarks. Involves substantial modifications that fundamentally transform the interpretation or message conveyed by the image. It may even appear surprising or strange to an audience. 

# Using the definitions of small, and large semantic changes above,"""

# output_struct = """

# Return the captions in the format:
# Original Caption:

# Modified Caption 1:
# Change 1:

# Modified Caption 2:
# Change 2:
# """


class LlavaGenerate_Edit:
    def __init__(
        self,
        args: argparse.Namespace,
        verbose: bool = False,
        prompt: str = "",
        image_file: str = "",
        cache_dir: str = "cache",
        model_name: str = "LlavaLlamaForCausalLM",
        model_path: str =  "liuhaotian/llava-v1.6-mistral-7b",
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
        # self.sizes ={"large":"Large Semantic Change","medium":"Medium Semantic Change" ,"small":"Small Semantic Change"}
        # self.sizes = {"small":"significantly and dramatically","large":"moderately and comparatively"}
        self.sizes = ["small","medium","large"]
        # self.sizes = ["small","large"]
        self.semantic_change = SentenceTransformer(sentence_embedding_model)
        self.verbose = verbose

    def _compute_embeddings(self,original_cap, perturbed_caps):
        sentences=[]
        sentences.append(original_cap)
        sentences.extend(perturbed_caps)
        embeddings = self.semantic_change.encode(sentences)
        return embeddings


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
            sem_change = 1-(util.pytorch_cos_sim(sem_embed[0],sem_embed[ix+1])).numpy()[0][0]
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

    def generate_prompt(self, size):

        output_struct = f"""
            Return the captions in the format:
            Original Caption:

            {size} semantic change 1:
            Caption for changed image 1:

            {size} semantic change 2:
            Caption for changed image 2:

            {size} semantic change 3:
            Caption for changed image 3:
            """
        prompt = context +  f" Write a caption for the image. Make 3 {size} semantic change in the image. Give the caption for the changed image as well"  + output_struct
        # prompt = context +  f" Write a caption for the image. Make 3 large semantic change in the image. Give the caption for the changed image as well"
        # prompt = context +  f"Write a detailed caption for the image. Give me 5 unique {size} changes I can make to the original caption." + output_struct
        return prompt 

    def check_caption(self,og):
        # pdb.set_trace()
        ai_chat ="A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's question"
        sem_embed =  self.semantic_change.encode([og, ai_chat])
        sim = util.pytorch_cos_sim(sem_embed[0],sem_embed[1]).numpy()[0][0]
        # print(sim)
        if(sim > 0.7):
            return False
        else:
            return True
        
    def get_captions(self, image_file,perturbation_type='all'):
        perturbations = []
        edit_type = []
 
        setattr(self.args,"image_file", image_file) 
        # pdb.set_trace()
        if perturbation_type == "all":
            perturbation_types = self.sizes
        else:
            perturbation_types = [perturbation_type]
        
        for perturbation_type in perturbation_types:
            modified_captions = []
            prompt = self.generate_prompt(perturbation_type)
            setattr(self.args,"query",prompt) 
            for i in range(5):
                responses = eval_model_wo_loading(self.tokenizer, self.model, self.image_processor, self.context_len, self.args)
                # original_caption_match = re.search(r'Original Caption:\s+(.*?)\n', responses, re.DOTALL)
                # original_caption[perturbation_type] = original_caption_match.group(1).strip()
                lines = responses.split('\n')
                
                try:
                    original_caption_index = lines.index('Original Caption:') + 1
                except:
                    original_caption_index = next(index for index, line in enumerate(lines) if line.startswith('Original Caption'))
                    # pdb.set_trace()
                original_caption_1 = lines[original_caption_index].replace('Original Caption:', '').strip()
                if(self.check_caption(original_caption_1)):
                    original_caption = original_caption_1
                    capture_changed_caption = False
                    for j, line in enumerate(lines):
                        line = line.strip()
                        if line.startswith('Caption for changed image'):
                            if(line.split(':')[-1] == ""):
                                try:
                                    modified_captions.append(lines[j+1].strip())
                                except:
                                    pdb.set_trace()
                            else:
                                modified_captions.append(line.split(':')[-1])
                    # modified_captions = []
                    # for match in re.finditer(r'Modified Caption (\d+):(.*?)Change \1:(.*?)\n\n', responses, re.DOTALL):
                    #     caption_number = match.group(1)
                    #     modified_caption = match.group(2).strip() + match.group(3).strip()
                    #     modified_captions.append(modified_caption)

                    # original_caption = re.search(self.original_caption_pattern, responses).group(1)
                    # modified_caption = re.findall(self.modified_caption_pattern, responses)
                    
                    perturbations.extend(modified_captions)
                    edit_type.extend([perturbation_type]*len(modified_captions))
                    break
                else:
                    continue
        # pdb.set_trace()    

        if (len(perturbations) == 0):
            perturbed_captions_dict_list = {"empty":"empty"}
            return "", perturbed_captions_dict_list
        perturbed_captions_dict_list = self._parse_perturbations(original_caption , perturbations,edit_type)
        random.shuffle(perturbed_captions_dict_list)
        if self.verbose:
            for ix in range(min(len(perturbed_captions_dict_list), 10)):
                ecap = perturbed_captions_dict_list[ix]["edited_caption"]
                logger.info(f"{ix+1}: {ecap}\n")
            logger.info(f"...\n")
        if self.verbose:
            logger.info(f"=> Generated caption: {original_caption}")
        
        return original_caption, perturbed_captions_dict_list


        

