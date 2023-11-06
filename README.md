# How-Fake-is-Too-Fake

## Repo Outline:

* `data_generation` --
* `model_arch` -- 
* `preprocessing` --
    * `create_repaint_input.py` -- Takes a dataset with comprehensive image and mask pairs. Selects a N masks per images, and prepares these pairs for input to RePaint (rough draft).
    * `currate_ade20k.py` -- Contains all preprocessing functions to resize and isolate image mask of the ADE20k dataset. Functions in this script can be adapted to other datasets as well. So long as file paths and metadata attributes are adapted appropriately.
* `stress_testing` -- 