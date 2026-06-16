# Données

Ce dossier doit contenir vos séquences RGB-D pour la reconstruction.

## Structure attendue

```
data/
├── color/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
└── depth/
    ├── 0001.png   (16-bit, en millimètres)
    ├── 0002.png
    └── ...
```

Les images couleur et de profondeur doivent porter le même nom et être alignées
pixel à pixel (même résolution, même point de vue).

## Sources de données possibles

### 1. Capturer vos propres données
- Smartphone avec capteur LiDAR (iPhone Pro via apps comme "Record3D" ou "Polycam")
- Caméra RGB-D (Intel RealSense, Azure Kinect, Microsoft Kinect)

### 2. Jeux de données publics (scènes urbaines / rue)
- **TUM RGB-D Dataset** : https://cvg.cit.tum.de/data/datasets/rgbd-dataset
- **KITTI** (LiDAR + caméra, scènes de rue) : https://www.cvlibs.net/datasets/kitti/
- **Cityscapes** (images urbaines annotées, pour tester la segmentation) : https://www.cityscapes-dataset.com/
- **ScanNet** (scènes intérieures RGB-D) : http://www.scan-net.org/

## Génération de profondeur sans capteur

Si vous n'avez que des images RGB (pas de profondeur), vous pouvez générer
des cartes de profondeur avec un modèle d'estimation monoculaire :

- **Depth Anything** : https://github.com/LiheYoung/Depth-Anything
- **MiDaS** : https://github.com/isl-org/MiDaS

Exemple rapide avec Depth Anything (à intégrer dans `src/`) :
```python
from transformers import pipeline
depth_estimator = pipeline("depth-estimation", model="LiheYoung/depth-anything-small-hf")
depth = depth_estimator(image)["depth"]
```

Notez que la profondeur estimée n'est pas à l'échelle métrique réelle ;
ajustez `DEPTH_SCALE` dans `src/config.py` en conséquence, ou normalisez
manuellement selon une référence connue dans la scène.

## Ajuster les paramètres caméra

Modifiez `DEFAULT_INTRINSICS` dans `src/config.py` selon votre capteur
(résolution, focales fx/fy, centre optique cx/cy).
