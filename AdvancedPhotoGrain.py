import torch
import torch.nn.functional as F
import comfy
import numpy as np
# Ajout nécessaire pour l'effet de flou
import torchvision.transforms.functional as TF

class PhotoFilmGrain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "grain_type": (
                    ["gaussian", "poisson", "perlin"], {"default": "poisson"}
                ),
                "grain_intensity": (
                    "FLOAT", {"default": 0.022, "min": 0.001, "max": 1.0, "step": 0.001}
                ),
                "grain_size": (
                    "FLOAT", {"default": 1.5, "min": 1.0, "max": 16.0, "step": 0.1}
                ),
                "saturation_mix": (
                    "FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01}
                ),
                "adaptive_grain": (
                    "FLOAT", {"default": 0.30, "min": 0.0, "max": 2.0, "step": 0.01}
                ),
                # NOUVEL EFFET "FILM LOOK"
                "halation_strength": (
                    "FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}
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
        self, images, grain_type, grain_intensity, grain_size, saturation_mix,
        adaptive_grain, halation_strength, vignette_strength, chromatic_aberration
    ):
        device = comfy.model_management.get_torch_device()
        images = images.to(device)

        if grain_intensity > 0.0:
            if grain_type == "gaussian":
                grain = self._generate_gaussian(images, grain_size)
            elif grain_type == "poisson":
                grain = self._generate_poisson(images, grain_size)
            elif grain_type == "perlin":
                grain = self._generate_perlin(images, grain_size)
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
        else:
            output = images

        if halation_strength > 0.0:
            output = self._apply_halation(output, halation_strength)

        if vignette_strength > 0.0:
            output = self._apply_vignette(output, vignette_strength)

        if chromatic_aberration > 0.0:
            output = self._apply_chromatic_aberration(output, chromatic_aberration)

        return (output.to(comfy.model_management.intermediate_device()),)

    def _generate_gaussian(self, images, size):
        B, H, W, C = images.shape
        size = int(size) 
        if size <= 1:
            noise = torch.randn_like(images)
        else:
            small_H, small_W = H // size, W // size
            noise = torch.randn(B, small_H, small_W, C, device=images.device)
            noise = noise.permute(0, 3, 1, 2)
            noise = F.interpolate(noise, size=(H, W), mode="nearest")
            noise = noise.permute(0, 2, 3, 1)
        
        noise[:, :, :, 0] *= 2.0
        noise[:, :, :, 2] *= 3.0
        return noise

    def _generate_poisson(self, images, size):
        B, H, W, C = images.shape
        size = int(size) 
        if size <= 1:
            target_images = images
        else:
            small_H, small_W = H // size, W // size
            target_images = F.interpolate(images.permute(0, 3, 1, 2), size=(small_H, small_W), mode="bilinear", align_corners=False)
            target_images = target_images.permute(0, 2, 3, 1)

        scaled = torch.clamp(target_images * 255.0, 0, 255).round()
        noise = torch.poisson(scaled) - scaled
        grain = (noise / 255.0) * 16.0

        if size > 1:
            grain = grain.permute(0, 3, 1, 2)
            grain = F.interpolate(grain, size=(H, W), mode="nearest")
            grain = grain.permute(0, 2, 3, 1)

        grain[:, :, :, 0] *= 2.0
        grain[:, :, :, 2] *= 3.0
        return grain

    def _generate_perlin(self, images, size):
        B, H, W, C = images.shape
        size = int(size)
        scale = max(4, 32 // size)
        
        perlin = self._make_fractal_noise(B, H, W, scale, images.device)
        perlin = (perlin - 0.5) * 2.0
        
        perlin = perlin.unsqueeze(3).repeat(1, 1, 1, 3)
        return perlin

    def _make_fractal_noise(self, B, H, W, scale, device, octaves=4, persistence=0.5, lacunarity=2.0):
        total_noise = torch.zeros(B, H, W, device=device)
        frequency = 1.0
        amplitude = 1.0
        max_amplitude = 0.0

        for _ in range(octaves):
            current_scale = max(2, int(scale * frequency))
            coarse_noise = torch.rand(B, current_scale, current_scale, 1, device=device)
            upscaled_noise = F.interpolate(coarse_noise.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
            total_noise += upscaled_noise * amplitude
            max_amplitude += amplitude
            amplitude *= persistence
            frequency *= lacunarity

        if max_amplitude > 0:
            total_noise /= max_amplitude
            
        return total_noise

    def _apply_halation(self, image, strength):
        B, H, W, C = image.shape
        if C != 3: return image

        # 1. Isoler les hautes lumières
        luma = image.mean(dim=3, keepdim=True)
        highlights_mask = torch.clamp((luma - 0.75) * 4, 0, 1) # Seuil doux à 0.75
        
        # 2. Créer le halo rouge en floutant le canal rouge des hautes lumières
        red_channel = image[:, :, :, 0:1]
        red_glow = red_channel * highlights_mask
        
        # Le rayon du flou dépend de la force et de la taille de l'image
        blur_radius = int(strength * (W / 25)) * 2 + 1
        if blur_radius < 3: return image

        # Permutation pour le flou : (B, H, W, C) -> (B, C, H, W)
        red_glow_permuted = red_glow.permute(0, 3, 1, 2)
        red_glow_blurred = TF.gaussian_blur(red_glow_permuted, kernel_size=blur_radius)
        # Retour à (B, H, W, C)
        red_glow_blurred = red_glow_blurred.permute(0, 2, 3, 1)

        # 3. Créer la couche de halation (0 pour vert/bleu, le flou pour rouge)
        halation_layer = torch.cat([
            red_glow_blurred, 
            torch.zeros_like(red_glow_blurred), 
            torch.zeros_like(red_glow_blurred)
        ], dim=3)

        # 4. Ajouter la halation à l'image
        return (image + halation_layer * strength).clamp(0.0, 1.0)

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

NODE_CLASS_MAPPINGS = {
    "PhotoFilmGrain": PhotoFilmGrain,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoFilmGrain": "📸 Photo Film Grain",
}