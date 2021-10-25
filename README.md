# Generative neural networks for seismic data

## Intro


## Installation

1. Install requirements & repo for baseline segmentation

```
git clone git@github.com:Miffka/seismogen.git
cd seismogen
pip install -r requirements/train.txt
pip install -r requirements/torch.txt
pip install -e .
```

2. Install requirements & repo for [MobileStyleGAN](https://github.com/bes-dev/MobileStyleGAN.pytorch)

I only made it a package.

```
git clone git@github.com:Miffka/MobileStyleGAN.pytorch.git
pip install -r requirements.txt
pip install -e .
```

## Data

Download data from [Google Drive Folder](https://drive.google.com/drive/folders/1Nc1geC-cxzO5sngKJXptNFk_-2CaPmcv?usp=sharing) and put it into `data/` folder in the root of repository.

## Experiments

### Baseline experiments

To train baseline models use command `python seismogen/models/hor_segmentation/train.py` with the following arguments

1. Train on both F3 Demo and Penobscot
```bash
python seismogen/models/hor_segmentation/train.py --augmentation_intensity slight --seg_model_arch FPN --pretrained_weights imagenet --epochs 5 --task_name e5_fpn_slight_tr_f3_pen --evaluate_before_training --train_datasets f3_demo penobscot
```

2. Train on F3 Demo, evaluate on Penobscot
```bash
python seismogen/models/hor_segmentation/train.py --augmentation_intensity slight --seg_model_arch FPN --pretrained_weights imagenet --epochs 5 --task_name e5_fpn_slight_tr_f3_te_pen --evaluate_before_training --train_datasets f3_demo --test_datasets penobscot
```

3. Train on Penobscot, evaluate on F3 Demo

```bash
python seismogen/models/hor_segmentation/train.py --augmentation_intensity slight --seg_model_arch FPN --pretrained_weights imagenet --epochs 5 --task_name e5_fpn_slight_tr_pen_te_f3 --evaluate_before_training --train_datasets penobscot --test_datasets f3_demo
```


## Additional info

Review of the data used in the work:

```json
{
  "Kerry": [
    {
      "volume": "raw/Kerry/Kerry3e.sgy",
      "horizons": "raw/Kerry/Kerry_h_ix_bulk.dat",
      "markup": "processed/Kerry/markup/00_Kerry3e.csv"
    }
  ],
  "Parihaka": [
    {
      "volume": "raw/Parihaka/Parihaka_PSTM_far_stack.sgy",
      "horizons": "raw/Parihaka/Parihaka_h_ix_bulk.dat",
      "markup": "processed/Parihaka/markup/00_Parihaka_PSTM_far_stack.csv"
    }
  ],
  "Poseidon": [
    {
      "volume": "raw/Poseidon/Poseidon_i1000-3600_x900-3200.sgy",
      "horizons": "raw/Poseidon/Poseidon_h_ix_bulk.dat",
      "markup": "processed/Poseidon/markup/00_Poseidon_i1000-3600_x900-3200.csv"
    }
  ],
  "SEG_2020_W_18": [
    {
      "volume": "raw/SEG_2020_W_18/TestData_Image1.segy",
      "markup": "processed/SEG_2020_W_18/markup/00_TestData_Image1.csv"
    },
    {
      "volume": "raw/SEG_2020_W_18/TestData_Image2.segy",
      "markup": "processed/SEG_2020_W_18/markup/01_TestData_Image2.csv"
    },
    {
      "volume": "raw/SEG_2020_W_18/TrainingData_Image.segy",
      "mask": "raw/SEG_2020_W_18/TrainingData_Labels.segy",
      "markup": "processed/SEG_2020_W_18/markup/02_TrainingData_Image.csv"
    }
  ],
  "f3_demo": [
    {
      "volume": "raw/f3_demo/f3_demo_2020_wnull.sgy",
      "horizons": "raw/f3_demo/f3_3d_horizons.dat",
      "markup": "processed/f3_demo/markup/00_f3_demo_2020_wnull.csv"
    }
  ],
  "FORCE_ML_Competition_2020": [
    {
      "volume": "raw/FORCE_ML_Competition_2020/ichthys_3D_seismic_for_fault_competition.sgy",
      "markup": "processed/FORCE_ML_Competition_2020/markup/00_ichthys_3D_seismic_for_fault_competition.csv"
    }
  ],
  "penobscot": [
    {
      "volume": "raw/penobscot/1-PSTM_stack_agc.sgy",
      "horizons": "raw/penobscot/penobscot_horizons.dat",
      "markup": "processed/penobscot/markup/00_1-PSTM_stack_agc.csv"
    }
  ]
}
```


(c) Miffka