import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

class CaptionGenerator:
    def __init__(
        self,
        device: torch.device,
        caption_model_type: Optional[str] = "Salesforce/blip-image-captioning-large",
        repetition_penalty=1.0,
        min_caption_length=20,
        max_caption_length=100,
        sentence_sim:'sentence-transformers/all-MiniLM-L6-v2'
    ):
        """
        Initialize caption generator
        Args:
            args: Command line arguments from argparse
            device: Device to run model on
            model_type: Type of model. Defaults to "large_coco".
            repetition_penalty: Repetition penalty. Defaults to 1.0.
        """
        # self.repetition_penalty = repetition_penalty
        # self.min_length = min_caption_length
        self.max_length = max_caption_length
        self.device = device
        self.caption_model_type = caption_model_type
        self.sentence_sim = sentence_sim

        self.vis_processor = BlipProcessor.from_pretrained(self.model_type)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(self.model_type).to(self.device)
        self.text = "a photo of"
        self.model_sem_sim = SentenceTransformer(self.sentence_sim)


    def _generate(self,img_path: str):
        """
        Generate caption for image
        Args:
            img: Image path to generate caption for
        Returns:
            gencap (str): Generated caption
        """

        img = Image.open(img_path).convert("RGB")
        inputs = self.vis_processor(img, self.text, max_length=self.max_length, return_tensors="pt")
        out = self.caption_model.generate(**inputs)
        gencap = self.vis_processor.decode(out[0], skip_special_tokens=True)
        return gencap

    def generate(self,real_img_path: str ,fake_img_path: str):
        """
        Generate caption for image
        Args:
            img: Image path to generate caption for
        Returns:
            gencap (str): Generated caption
        """

        real_gencap = self._generate(real_img_path)
        fake_gencap = self._generate(fake_img_path)
        score = self.cos_sim(real_gencap, fake_gencap)
        return real_gencap, fake_gencap, score

    def cos_sim(real_caption, fake_caption):
        sentences = [real_caption, fake_caption]
        embeddings = self.model_sem_sim.encode(sentences)
        val = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        sem_sim_caption = val.numpy()[0][0]
        
    return sem_sim_caption