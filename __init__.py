import torch
import numpy
from PIL import Image

class ImageFillBackgroundColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "color": ("COLOR", {"default": "#FFFFFF"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "process"
    CATEGORY = "ConForUI"

    def hex_to_rgb(self, hex_color):
        # Remove hashtag and trans hex to RGB tuple
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def process(self, input_image, color):
        # Get RGB
        rgb_color = self.hex_to_rgb(color)
        results = []

        for img_tensor in input_image:
            # Trans input image from Tensor [H, W, C] to Pillow format
            i = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(numpy.clip(i, 0, 255).astype(numpy.uint8)).convert("RGBA")

            # Build a pure color background
            background = Image.new("RGBA", img.size, rgb_color + (255,))

            # Combine to new image
            combined = Image.alpha_composite(background, img).convert("RGB")

            # Trans back to Tensor [H, W, C] format
            result_tensor = torch.from_numpy(numpy.array(combined).astype(numpy.float32) / 255.0)
            results.append(result_tensor)

        return (torch.stack(results, dim=0),)

NODE_CLASS_MAPPINGS = {"ImageFillBackgroundColor": ImageFillBackgroundColor}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageFillBackgroundColor": "Image Add Background Color (Hex)"}