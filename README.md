# Advanced-Photo-Grain ComfyUI Node
<img width="855" height="556" alt="Capture d'écran 2025-07-19 011317" src="https://github.com/user-attachments/assets/9d091c57-96d5-4d4d-b921-b38e0e639a4f" />

---

## 🎛️ Paramètres du Node

| Paramètre | Type | Description |
|---|---|---|
| **grain\_type** | `gaussian` / `poisson` / `perlin` | Le type de bruit utilisé :\<br\> - *Gaussian* : bruit blanc classique, réaliste\<br\> - *Poisson* : bruit simulant le comportement de photons (effet caméra, discret)\<br\> - *Perlin* : bruit texturé à grande échelle, type film 35mm |
| **grain\_intensity** | `FLOAT` (0.01 – 1.0) | Intensité globale du grain ajouté à l’image. Valeur typique : `0.025`. Poisson est plus subtil, nécessite souvent `1.0`. |
| **grain\_size** | `FLOAT` (1.0 – 16.0) | Contrôle la taille du grain. `1.0` est un grain très fin (niveau pixel). Des valeurs plus élevées créent des amas de grain plus grossiers, simulant différentes pellicules ou sensibilités ISO. |
| **saturation\_mix** | `FLOAT` (0.0 – 1.0) | Mélange entre bruit coloré (1.0) et bruit monochrome (0.0). Valeur typique : `0.4` à `0.5` pour un rendu cinéma. |
| **adaptive\_grain** | `FLOAT` (0.0 – 2.0) | Amplifie le grain selon la luminosité de l’image (plus de grain dans les ombres). `0.0` = désactivé. `0.8` à `1.5` donne de bons résultats en portrait. |
| **halation\_strength**| `FLOAT` (0.0 – 1.0) | Simule l'effet de halo rouge autour des hautes lumières, typique des pellicules argentiques. Visible à partir de `0.2`. |
| **lens\_distortion**| `FLOAT` (-0.5 – 0.5) | Simule la distorsion optique de l'objectif. Négatif = barillet (convexe), Positif = coussinet (concave). |
| **vignette\_strength** | `FLOAT` (0.0 – 1.0) | Assombrit légèrement les bords pour simuler un vignettage optique. `0.15` recommandé. |
| **chromatic\_aberration** | `FLOAT` (0.0 – 5.0) | Ajoute une frange RGB aux contours, en simulant l’aberration chromatique des objectifs photo. Effet visible à partir de `0.5`. Typiquement `0.3` à `0.8`. |

---
