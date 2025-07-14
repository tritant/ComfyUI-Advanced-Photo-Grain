# ComfyUI-Advanced-Photo-Grain
<img width="515" height="385" alt="Capture d'écran 2025-07-14 120943" src="https://github.com/user-attachments/assets/2e39857b-63da-4604-9a67-ee6d35ec3a54" />

---

## 🎛️ Paramètres du Node

| Paramètre | Type | Description |
|----------|------|-------------|
| **grain_type** | `gaussian` / `poisson` / `perlin` | Le type de bruit utilisé :<br> - *Gaussian* : bruit blanc classique, réaliste<br> - *Poisson* : bruit simulant le comportement de photons (effet caméra, discret)<br> - *Perlin* : bruit texturé à grande échelle, type film 35mm |
| **grain_intensity** | `FLOAT` (0.01 – 1.0) | Intensité globale du grain ajouté à l’image. Valeur typique : `0.025`. Poisson est plus subtil, nécessite souvent `1.0`. |
| **saturation_mix** | `FLOAT` (0.0 – 1.0) | Mélange entre bruit coloré (1.0) et bruit monochrome (0.0). Valeur typique : `0.4` à `0.5` pour un rendu cinéma. |
| **adaptive_grain** | `FLOAT` (0.0 – 2.0) | Amplifie le grain selon la luminosité de l’image (plus de grain dans les ombres). `0.0` = désactivé. `0.8` à `1.5` donne de bons résultats en portrait. |
| **vignette_strength** | `FLOAT` (0.0 – 1.0) | Assombrit légèrement les bords pour simuler un vignettage optique. `0.15` recommandé. |
| **chromatic_aberration** | `FLOAT` (0.0 – 5.0) | Ajoute une frange RGB aux contours, en simulant l’aberration chromatique des objectifs photo. Effet visible à partir de `0.5`. Typiquement `0.3` à `0.8`. |

---
