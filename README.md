# How-Fake-is-Too-Fake

## Repo Outline:

* `data_generation` --
* `model_arch` -- 
* `preprocessing` --
    * `create_repaint_input.py` -- Takes a dataset with comprehensive image and mask pairs. Selects a N masks per images, and prepares these pairs for input to RePaint (rough draft).
    * `currate_ade20k.py` -- Contains all preprocessing functions to resize and isolate image mask of the ADE20k dataset. Functions in this script can be adapted to other datasets as well. So long as file paths and metadata attributes are adapted appropriately.
    * `currate_HPD.py` -- This script processes the Human Parsing Dataset. Contains a helpful function to check if a mask was clipped during crop and scaling down to 256 * 256.
* `stress_testing` -- 

## Llava Guided generation

The jupyter notebook containing the code to generate the semantic and size based change is "llava_captions.ipynb" in the llava_code folder

Llava weights are in the folder ``llava_weights''. Add this folder as a path wherever Llava weights are asked in the file "llava_captions.ipynb"

Download the llava github repository and do the following steps:

- Replace llava/eval/run_llava.py with the file in llava_code
- Replace llava/model/builder.py with the file in llava_code
- Copy llava_captions.ipynb into the LLaVA git cloned folder

You are now ready to run the code!
