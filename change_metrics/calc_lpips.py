import torch
from PIL import Image
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class LPIPS:
    def __init__(
        self,
        net_type: str = 'vgg',
        normalize=True,
        reduction='mean',
    ):
        """
        Initialize caption generator
        Args:
            args: Command line arguments from argparse
            net_type: str indicating backbone network type to use.
            normalize: by default this is False meaning that the input is expected to be in the [-1,1] range. If set to True will instead expect input to be in the [0,1] range.
            reduction: str indicating how to reduce over the batch dimension. Choose between ‘sum’ or ‘mean’.
        """

        
        self.net_type = net_type
        self.normalize = normalize
        self.reduction = reduction
        self.model = LearnedPerceptualImagePatchSimilarity(net_type=self.net_type,normalize=self.normalize, reduction=self.reduction)
        


    def calc_lpips(self, img_real_rgb, img_fake_rgb):
        """
        Generate caption for image
        Args:
            img: Image path to generate caption for
        Returns: 
            gencap (str): Generated caption
        """
        fake_img = self.normalize_img(img_fake_rgb).permute(2,0,1).unsqueeze(0)
        real_img = self.normalize_img(img_real_rgb).permute(2,0,1).unsqueeze(0)
        score = self.model(real_img, fake_img)

        return score

    def normalize_img(arr):

        if(self.normalize):

            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

            transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std) 
            ]
            arr_new = transform(arr)
        else:
            arr_min = arr.min()
            arr_max = arr.max()
            arr_range = arr_max - arr_min
            scaled = np.array((arr-arr_min) / float(arr_range), dtype='f')
            arr_new = -1 + (scaled * 2)

        return arr_new