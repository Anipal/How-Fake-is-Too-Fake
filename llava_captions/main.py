# Copyright 2023 the LANCE team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

import os
import json
import logging
import cv2
from tqdm import tqdm
import argparse
import torch
from PIL import Image
import pdb
import torchvision.datasets as datasets
import numpy as np
# from accelerate import Accelerator

# accelerator = Accelerator()
# from accelerate.logging import get_logger
# from lance.generate_captions import *
# from lance.edit_captions import *
# from lance.edit_images import *
# from lance.utils.misc_utils import *
from edit_and_gen_captions_llava import *


def main(args: argparse.Namespace):
    # logging.info(accelerator.state, main_process_only=True)
    # device = accelerator.device
    # if args.verbose:
    #     logger = get_logger("lance")
    #     for arg, value in sorted(vars(args).items()):
    #         logger.debug("{}: {}", arg, value)
    #     logger.info("------------------------------------------------")
    #     logger.info(f"=> Initializing LANCE")
    if args.dset_name == "HardImageNet":
        import datasets.hard_imagenet as ha

        dset = ha.HardImageNet(args.img_dir)
        # if args.verbose:
        #     logger.info(f"=> Loaded dataset from {args.img_dir}")
    elif args.dset_name == "ImageFolder":
        import datasets.custom_imagefolder as cif

        dset = cif.CustomImageFolder(args.img_dir, args.json_path)
    #     if args.verbose:
    #         logger.info(f"=> Loaded dataset from {args.img_dir}")
    # else:
    #     logger.error("Dataset type not supported, exiting")
    #     raise ValueError("Dataset not supported")

    data_sampler = torch.utils.data.sampler.SequentialSampler(dset)

    dataloader = torch.utils.data.DataLoader(
        dset,
        batch_size=1,
        shuffle=(data_sampler is None),
        num_workers=6,
        pin_memory=True,
        sampler=data_sampler,
        drop_last=True,
    )

    editcap_dict = {}

    if args.load_caption_edits:
        if not os.path.exists(args.editcap_dict_path):
            raise ValueError("Path to edited caption file does not exist")
        editcap_dict = json.load(open(args.editcap_dict_path, "r"))


    if not args.load_caption_edits:
        caption_editor = LlavaGenerate_Edit(
            args,verbose=args.verbose
        )

    # model = image_editor.model
    # dataloader, model = accelerator.prepare(dataloader, model)
    
    for paths, targets in dataloader:
        # Generate caption
        img_path, clsname = paths[0], targets[0]
        if len(np.array(Image.open(img_path)).shape) < 3:
            continue  # Ignore grayscale images
        out_dir = os.path.join(args.lance_path, args.exp_id)
        os.makedirs(out_dir, exist_ok=True)
        if(not(os.path.exists(os.path.join(out_dir,'edited_captions',args.perturbation_type)))):
            os.makedirs(os.path.join(out_dir,'edited_captions',args.perturbation_type))
        # if args.verbose:
        #     logger.info(f"=>Generating LANCE for {img_path}")

        img_name = img_path.split("/")[-1]

        # Edit caption
        if img_name in editcap_dict.keys():
            # if args.verbose:
            #     logger.warning(
            #         "Caption edits already generated, loading from dictionary\n"
            #     )
            new_caps = editcap_dict[img_name]
            cap = editcap_dict[img_name]['original_caption']
        else:
            # pdb.set_trace()
            cap, new_caps = caption_editor.get_captions(
               img_path, args.perturbation_type 
            )
            editcap_dict[img_name] = new_caps
            save_path = os.path.join(out_dir,'edited_captions',args.perturbation_type,(img_name.split('.')[0]+'.json'))
            json.dump(new_caps, open(save_path, 'w'), indent=4)
            img = cv2.imread(img_path)
            img_2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            save_img_path = os.path.join(out_dir,img_name)
            cv2.imwrite(save_img_path,img_2)

        out_path = os.path.join(out_dir, os.path.splitext(img_name)[0])
        if os.path.exists(out_path):
            # if args.verbose:
            #     logger.warning(f"=> Image `{out_path}' already edited, skipping")
            continue
        

        # # Invert image
        # _, _, x_t, uncond_embeddings = image_editor.invert(img_path, cap, out_dir)

        # # Edit image
        # image_editor.edit(
        #     out_path, clsname.lower(), x_t, uncond_embeddings, cap, new_caps
        # )
        # del x_t, uncond_embeddings
        # accelerator.free_memory()
    save_path_edited = os.path.join(out_dir,'edited_captions', json_path.split('/')[-1])
    json.dump(new_caps, open(save_path_edited, 'w'), indent=4)
    json.dump(vars(args), open(out_dir + "/args.json", "w"), indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ###########################################################################
    # Experiment identifier
    ###########################################################################
    parser.add_argument("--exp_id", type=str, default="lance")
    parser.add_argument(
        "--dset_name",
        type=str,
        help="Dataset name: HardImageNet or ImageFolder",
        default="HardImageNet",
    )
    parser.add_argument("--img_dir", type=str, help="ImageFolder containing images")
    parser.add_argument("--json_path", type=str, default = "outputs",help="Path to input metadata json")
    parser.add_argument("--csv", type=str, default = "outputs",help="Path to csv containing image list")
    parser.add_argument(
        "--lance_path",
        type=str,
        default="outputs",
        help="LANCE output directory",
    )
    ###########################################################################
    # Caption generator hyperparameters
    ###########################################################################
    parser.add_argument("--gencaption_name",
        type=str,
        help="Captioner name: blip2_opt,blip2_t5,blip2,blip_caption",
        default="blip_caption",
    )
    parser.add_argument(
        "--load_captions", action="store_true", help="Load captions from path"
    )
    parser.add_argument(
        "--gencap_dict_path",
        type=str,
        default="outputs/hard_imagenet_captions_blip2.json",
        help="Path to JSON file containing image captions",
    )
    parser.add_argument(
        "--load_caption_edits", action="store_true", help="Load captions from path"
    )
    parser.add_argument(
        "--editcap_dict_path",
        type=str,
        default="outputs/hard_imagenet_captions_blip2_edited.json",
        help="Path to JSON file containing edited captions",
    )
    ###########################################################################
    # Caption editor hyperparameters
    ###########################################################################
    parser.add_argument(
        "--llama_finetuned_path",
        type=str,
        default="checkpoints/caption_editing/lit-llama-lora-finetuned.pth",
        help="Path to finetuning llama model in lightning format",
    )
    parser.add_argument(
        "--llama_pretrained_path",
        type=str,
        default="checkpoints/caption_editing/lit-llama.pth",
        help="Path to pretrained llama model in lightning format",
    )
    parser.add_argument(
        "--llama_tokenizer_path",
        type=str,
        default="checkpoints/caption_editing/tokenizer.model",
        help="Path to LLAMA tokenizer model",
    )
    parser.add_argument(
        "--perturbation_type",
        type=str,
        default="all",
        help="Type of perturbation to stress-test against",
    )
    ###########################################################################
    # Image editing hyperparameters
    ###########################################################################
    parser.add_argument(
        "--ldm_type",
        type=str,
        default="stable_diffusion_v1_4",
        help="Latent Diffusion Model to use",
    )
    parser.add_argument(
        "--text_similarity_threshold",
        type=float,
        default=0.5,
        help="Threshold for CLIP text similarity between GT class and word(s) being edited",
    )
    parser.add_argument(
        "--clip_img_thresh",
        type=float,
        default=0.0,
        help="Threshold for CLIP similarity between original and edited image",
    )
    parser.add_argument(
        "--clip_dir_thresh",
        type=float,
        default=0.0,
        help="Threshold for CLIP similarity between original and edited direction",
    )
    parser.add_argument(
        "--clip_thresh",
        type=float,
        default=0.0,
        help="Threshold for CLIP similarity between original and edited image and direction",
    )
    parser.add_argument(
        "--edit_word_weight",
        type=float,
        default=2.0,
        help="Maximum number of tries for editing a caption",
    )
    parser.add_argument(
        "--save_inversion",
        action="store_false",
        help="Whether to save image inversion and load from it for future edits",
    )
    parser.add_argument(
        "--verbose",
        action="store_false",
        help="Logging verbosity",
    )
    ###########################################################################
    args = parser.parse_args()
    main(args)
