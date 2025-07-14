import torch
import torch.nn.functional as F
import comfy
import numpy as np

class PhotoFilmGrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "grain_type": (
                    ["gaussian", "poisson", "perlin"], {"default": "gaussian"}
                ),
                "grain_intensity": (
                    "FLOAT", {"default": 0.012, "min": 0.01, "max": 1.0, "step": 0.001}
                ),
                "saturation_mix": (
                    "FLOAT", {"default": 0.52, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "adaptive_grain": (
                    "FLOAT", {"default": 0.50, "min": 0.0, "max": 2.0, "step": 0.01}
                ),
                "vignette_strength": (
                    "FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "chromatic_aberration": (
                    "FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.1}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_grain"
    CATEGORY = "image/enhancement"
    DESCRIPTION = "Adds realistic film grain, vignette and RGB aberration to photos"

    def apply_grain(
        self, images, grain_type, grain_intensity, saturation_mix,
        adaptive_grain, vignette_strength, chromatic_aberration
    ):
        device = comfy.model_management.get_torch_device()
        images = images.to(device)

        if grain_type == "gaussian":
            grain = self._generate_gaussian(images)
        elif grain_type == "poisson":
            grain = self._generate_poisson(images)
        elif grain_type == "perlin":
            grain = self._generate_perlin(images)
        else:
            raise ValueError(f"Unsupported grain type: {grain_type}")

        gray = grain[:, :, :, 1].unsqueeze(3).repeat(1, 1, 1, 3)
        grain = saturation_mix * grain + (1.0 - saturation_mix) * gray

        if adaptive_grain > 0.0:
            luma = images.mean(dim=3, keepdim=True)
            gain = (1.0 - luma).pow(2.0) * 2.5
            grain = grain * (1.0 + adaptive_grain * gain)

        output = images + grain * grain_intensity
        output = output.clamp(0.0, 1.0)

        if vignette_strength > 0.0:
            output = self._apply_vignette(output, vignette_strength)

        if chromatic_aberration > 0.0:
            output = self._apply_chromatic_aberration(output, chromatic_aberration)

        return (output.to(comfy.model_management.intermediate_device()),)

    def _generate_gaussian(self, images):
        grain = torch.randn_like(images)
        grain[:, :, :, 0] *= 2.0
        grain[:, :, :, 2] *= 3.0
        return grain

    def _generate_poisson(self, images):
        scaled = torch.clamp(images * 255.0, 0, 255).round()
        noise = torch.poisson(scaled) - scaled
        grain = (noise / 255.0) * 16.0
        grain[:, :, :, 0] *= 2.0
        grain[:, :, :, 2] *= 3.0
        return grain

    def _generate_perlin(self, images):
        B, H, W, C = images.shape
        scale = 32
        perlin = self._make_perlin_noise(B, H, W, scale, images.device)
        perlin = perlin.unsqueeze(3).repeat(1, 1, 1, 3)
        return perlin

    def _make_perlin_noise(self, B, H, W, scale, device):
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing="ij"
        )
        coarse = torch.rand(B, scale, scale, 1, device=device)
        upscaled = F.interpolate(coarse.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False)
        return upscaled.squeeze(1)

    def _apply_vignette(self, image, strength):
        B, H, W, C = image.shape
        y = torch.linspace(-1, 1, H, device=image.device).view(1, H, 1, 1)
        x = torch.linspace(-1, 1, W, device=image.device).view(1, 1, W, 1)
        dist = torch.sqrt(x**2 + y**2)
        mask = 1.0 - strength * dist.clamp(0.0, 1.0)
        return image * mask

    def _apply_chromatic_aberration(self, image, strength):
        B, H, W, C = image.shape
        if C != 3 or strength <= 0:
            return image

        shift = max(1, round(strength * 2.5))

        r = F.pad(image[:, :, :, 0:1], (0, 0, shift, shift), mode='reflect')
        b = F.pad(image[:, :, :, 2:3], (0, 0, shift, shift), mode='reflect')
        g = image[:, :, :, 1:2]

        r = r[:, shift:H+shift, :, :]
        b = b[:, shift-1:H+shift-1, :, :]

        min_H = min(r.shape[1], g.shape[1], b.shape[1])
        min_W = min(r.shape[2], g.shape[2], b.shape[2])

        r = r[:, :min_H, :min_W, :]
        g = g[:, :min_H, :min_W, :]
        b = b[:, :min_H, :min_W, :]

        return torch.cat([r, g, b], dim=3)


# ComfyUI Registration
NODE_CLASS_MAPPINGS = {
    "PhotoFilmGrain": PhotoFilmGrain,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoFilmGrain": "📸 Photo Film Grain",
}
