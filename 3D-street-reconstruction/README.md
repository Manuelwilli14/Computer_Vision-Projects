# Reconstruction 3D + Segmentation Sémantique d'une Scène de Rue

Pipeline complet combinant **vision par ordinateur**, **deep learning** et **géométrie 3D** pour reconstruire une scène de rue à partir d'images et y identifier sémantiquement routes, voitures, bâtiments et végétation.

## Objectifs

- Reconstruire un nuage de points 3D d'une rue à partir de plusieurs images (Structure from Motion / RGB-D)
- Segmenter sémantiquement chaque image (routes, voitures, bâtiments, végétation, ciel...)
- Projeter les masques de segmentation 2D sur le nuage de points 3D pour obtenir une **scène 3D sémantiquement annotée**
- Visualiser le résultat avec Open3D

## Stack technique

| Composant | Rôle |
|---|---|
| **Open3D** | Reconstruction 3D, nuages de points, maillage, visualisation |
| **PyTorch** | Backend deep learning |
| **Segment Anything (SAM)** | Segmentation d'instances/masques |
| **DeepLabV3 / Cityscapes (torchvision)** | Segmentation sémantique avec classes prédéfinies (route, voiture, bâtiment, végétation) |
| **OpenCV** | Traitement d'images, calibration caméra |

## Architecture du pipeline

```
images RGB-D / multi-vue
        │
        ├──► [1] Segmentation sémantique 2D (DeepLabV3 + SAM)
        │         → masques : route, voiture, bâtiment, végétation, ciel...
        │
        ├──► [2] Reconstruction 3D (Open3D)
        │         → nuage de points / maillage via RGB-D ou SfM
        │
        └──► [3] Fusion 2D→3D
                  → chaque point 3D reçoit une étiquette sémantique
                  → nuage de points coloré par classe
                  → export .ply visualisable
```

## Structure du projet

```
3d-street-reconstruction/
├── README.md
├── requirements.txt
├── data/
│   └── README.md          # comment obtenir/placer les données
├── src/
│   ├── config.py
│   ├── semantic_segmentation.py   # DeepLabV3 (Cityscapes) + SAM
│   ├── reconstruction.py          # Open3D RGB-D / SfM → nuage de points
│   ├── fusion_2d_3d.py            # projection des labels sur le nuage 3D
│   ├── visualize.py               # affichage Open3D
│   └── pipeline.py                # orchestration complète
├── notebooks/
│   └── demo.ipynb
└── outputs/
    └── .gitkeep
```

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Téléchargez le checkpoint SAM (modèle ViT-B, ~375 Mo) :
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P checkpoints/
```

## Utilisation rapide

```bash
# 1. Segmentation sémantique d'une image
python src/semantic_segmentation.py --image data/sample_street.jpg --output outputs/seg_mask.png

# 2. Reconstruction 3D à partir de RGB-D (couleur + profondeur)
python src/reconstruction.py --color data/color/ --depth data/depth/ --output outputs/scene.ply

# 3. Fusion sémantique 2D → 3D
python src/fusion_2d_3d.py --pointcloud outputs/scene.ply --seg_dir outputs/segmentations/ --output outputs/scene_semantic.ply

# 4. Pipeline complet
python src/pipeline.py --data_dir data/ --output_dir outputs/

# 5. Visualisation
python src/visualize.py outputs/scene_semantic.ply
```

## Classes sémantiques détectées

Basé sur les classes Cityscapes (regroupées) :

| Classe | Couleur RVB |
|---|---|
| Route | (128, 64, 128) |
| Voiture | (0, 0, 142) |
| Bâtiment | (70, 70, 70) |
| Végétation | (107, 142, 35) |
| Ciel | (70, 130, 180) |
| Trottoir | (244, 35, 232) |
| Autre | (0, 0, 0) |

## Données de test

Le dossier `data/` contient des instructions pour récupérer des séquences RGB-D publiques (ex : KITTI, TUM RGB-D, ou vos propres photos avec un capteur de profondeur / smartphone LiDAR).

## Limitations & pistes d'amélioration

- La reconstruction RGB-D nécessite des images de profondeur alignées (capteur LiDAR/Kinect, ou estimation de profondeur monoculaire via MiDaS/Depth Anything)
- Pour une reconstruction sans capteur de profondeur, intégrer une étape SfM/MVS (ex : COLMAP) avant Open3D
- SAM segmente sans étiquettes sémantiques par défaut : ce projet combine SAM (masques précis) avec DeepLabV3 (étiquettes de classes) pour obtenir des masques sémantiques fiables
- Affiner avec un modèle entraîné spécifiquement sur Cityscapes/Mapillary pour de meilleures performances en environnement urbain

## Licence

MIT
