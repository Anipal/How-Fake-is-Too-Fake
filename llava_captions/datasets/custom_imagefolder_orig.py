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

import argparse
import torchvision.datasets as datasets

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, data_dir):
        super(CustomImageFolder, self).__init__(data_dir)
        self.data_dir = data_dir
        self.idx_to_class = { v: k for k, v in self.class_to_idx.items() }

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path_full, tgt = self.imgs[idx]
        return img_path_full, self.idx_to_class[tgt]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default="checkpoints/iminigenet",
        help="Path to custom dataset",
    )
    args = parser.parse_args()
    dset = CustomImageFolder(args.img_dir)
    print(dset[0])