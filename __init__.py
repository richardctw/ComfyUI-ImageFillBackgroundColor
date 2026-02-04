import torch
import numpy
from PIL import Image, ImageOps

class ImageFillBackgroundColor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "mask": ("MASK",),
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

    def process(self, input_image, mask, color):
        # Get RGB
        rgb_color = self.hex_to_rgb(color)
        results = []

        for i in range(len(input_image)):
            img_tensor = input_image[i]
            mask_tensor = mask[i] if i < len(mask) else mask[0]

            # Trans input image from Tensor [H, W, C] to Pillow format(RGBA)
            img_np = (255. * img_tensor.cpu().numpy()).astype(numpy.uint8)
            img_pil = Image.fromarray(img_np).convert("RGB")

            mask_np = (255. * mask_tensor.cpu().numpy()).astype(numpy.uint8)
            mask_pil = Image.fromarray(mask_np, mode='L')
            mask_pil = ImageOps.invert(mask_pil)
            if mask_pil.size != img_pil.size:
                mask_pil = mask_pil.resize(img_pil.size, resample=Image.BILINEAR)

            img_rgba = img_pil.copy()
            img_rgba.putalpha(mask_pil)

            # Build a pure color background
            background = Image.new("RGBA", img_rgba.size, rgb_color + (255,))

            # Combine to new image
            combined = Image.alpha_composite(background, img_rgba).convert("RGB")

            # Trans back to Tensor [H, W, C] format
            result_tensor = torch.from_numpy(numpy.array(combined).astype(numpy.float32) / 255.0)
            results.append(result_tensor)

        return (torch.stack(results, dim=0),)

NODE_CLASS_MAPPINGS = {"ImageFillBackgroundColor": ImageFillBackgroundColor}
NODE_DISPLAY_NAME_MAPPINGS = {"ImageFillBackgroundColor": "Image Add Background Color (Hex)"}