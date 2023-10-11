# EDA: Explicit Text-Decoupling and Dense Alignment for 3D Visual Grounding (CVPR2023)

> By [Yanmin Wu](https://yanmin-wu.github.io/), Xinhua Cheng, [Renrui Zhang](https://github.com/ZrrSkywalker), Zesen Cheng, [Jian Zhang*](http://villa.jianzhang.tech/)   
This repo is the official implementation of "**EDA: Explicit Text-Decoupling and Dense Alignment for 3D Visual Grounding**". [CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_EDA_Explicit_Text-Decoupling_and_Dense_Alignment_for_3D_Visual_Grounding_CVPR_2023_paper.html) | [arXiv](https://arxiv.org/abs/2209.14941) | [Code](https://github.com/yanmin-wu/EDA)

<figure>
<p align="center" >
<img src='./data/fig1.png' width=700 alt="Figure 1"/>
</p>
</figure>

## 0. Installation

+ **(1)** Install environment with `environment.yml` file:
  ```
  conda env create -f environment.yml --name EDA
  ```
  + or you can install manually:
    ```
    conda create -n EDA python=3.7
    conda activate EDA
    conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
    pip install numpy ipython psutil traitlets transformers termcolor ipdb scipy tensorboardX h5py wandb plyfile tabulate
    ```
+ **(2)** Install spacy for text parsing
  ```
  pip install spacy
  # 3.3.0
  pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.3.0/en_core_web_sm-3.3.0.tar.gz
  ```
+ **(3)** Compile pointnet++
  ```
  cd ~/EDA
  sh init.sh
  ```

## 1. [TODO] Quick visualization demo 
+ [ ] Visualization
+ [ ] Text-decoupling demo

## 2. Data preparation

The final required files are as follows:
```
├── [DATA_ROOT]
│	├── [1] train_v3scans.pkl # Packaged ScanNet training set
│	├── [2] val_v3scans.pkl   # Packaged ScanNet validation set
│	├── [3] ScanRefer/        # ScanRefer utterance data
│	│	│	├── ScanRefer_filtered_train.json
│	│	│	├── ScanRefer_filtered_val.json
│	│	│	└── ...
│	├── [4] ReferIt3D/        # NR3D/SR3D utterance data
│	│	│	├── nr3d.csv
│	│	│	├── sr3d.csv
│	│	│	└── ...
│	├── [5] group_free_pred_bboxes/  # detected boxes (optional)
│	├── [6] gf_detector_l6o256.pth   # pointnet++ checkpoint (optional)
│	├── [7] roberta-base/     # roberta pretrained language model
│	├── [8] checkpoints/      # EDA pretrained models
```

+ **[1] [2] Prepare ScanNet Point Clouds Data**
  + **1)** Download ScanNet v2 data. Follow the [ScanNet instructions](https://github.com/ScanNet/ScanNet) to apply for dataset permission, and you will get the official download script `download-scannet.py`. Then use the following command to download the necessary files:
    ```
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.ply
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.labels.ply
    python2 download-scannet.py -o [SCANNET_PATH] --type .aggregation.json
    python2 download-scannet.py -o [SCANNET_PATH] --type _vh_clean_2.0.010000.segs.json
    python2 download-scannet.py -o [SCANNET_PATH] --type .txt
    ```
    where `[SCANNET_PATH]` is the output folder. The scannet dataset structure should look like below:
    ```
    ├── [SCANNET_PATH]
    │   ├── scans
    │   │   ├── scene0000_00
    │   │   │   ├── scene0000_00.txt
    │   │   │   ├── scene0000_00.aggregation.json
    │   │   │   ├── scene0000_00_vh_clean_2.ply
    │   │   │   ├── scene0000_00_vh_clean_2.labels.ply
    │   │   │   ├── scene0000_00_vh_clean_2.0.010000.segs.json
    │   │   ├── scene.......
    ```
  + **2)** Package the above files into two .pkl files(`train_v3scans.pkl` and `val_v3scans.pkl`):
    ```
    python Pack_scan_files.py --scannet_data [SCANNET_PATH] --data_root [DATA_ROOT]
    ```
+ **[3] ScanRefer**: Download ScanRefer annotations following the instructions [HERE](https://github.com/daveredrum/ScanRefer). Unzip inside `[DATA_ROOT]`.
+ **[4] ReferIt3D**: Download ReferIt3D annotations following the instructions [HERE](https://github.com/referit3d/referit3d). Unzip inside `[DATA_ROOT]`.
+ **[5] group_free_pred_bboxes**: Download [object detector's outputs](https://1drv.ms/u/s!AsnjK0KGPk10gYBjpUjJm7TkADS8vg?e=1AXJdR). Unzip inside `[DATA_ROOT]`. (not used in single-stage method)
+ **[6] gf_detector_l6o256.pth**: Download PointNet++ [checkpoint](https://1drv.ms/u/s!AsnjK0KGPk10gYBXZWDnWle7SvCNBg?e=SNyUK8) into `[DATA_ROOT]`.
+ **[7] roberta-base**: Download the roberta pytorch model:
  ```
  cd [DATA_ROOT]
  git clone https://huggingface.co/roberta-base
  cd roberta-base
  rm -rf pytorch_model.bin
  wget https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin
  ```
+ **[8] checkpoints**: Our pre-trained models (see next step).

## 3. Models

|Dataset  | mAP@0.25 | mAP@0.5 | Model | Log (train) | Log (test)
|:---:|:---:|:---:|:---:|:---:|:---:|
|ScanRefer| 54.59 | 42.26 |[OneDrive](https://1drv.ms/u/s!AsnjK0KGPk10gYBa4hc26m5ZFkVPZw?e=zPN55r)*| [54_59.txt](https://1drv.ms/t/s!AsnjK0KGPk10gYBebAdozXnOgmm1YQ?e=H787s9)<sup>1</sup> / [54_44.txt](https://1drv.ms/t/s!AsnjK0KGPk10gYo27zspU40yhrF09A?e=8eRW6V)<sup>2</sup> | [log.txt](https://1drv.ms/t/s!AsnjK0KGPk10gYo-0fAXOoU1_sS6Bw?e=z8ANiN)
|ScanRefer (Single-Stage)| 53.83 | 41.70 |[OneDrive](https://1drv.ms/u/s!AsnjK0KGPk10gYBbGKhHSJXohqyruQ?e=oDFmSq)| [53_83.txt](https://1drv.ms/t/s!AsnjK0KGPk10gYBgx7E7P0NTBwOegQ?e=jdpEdp)<sup>1</sup> / [53_47.txt](https://1drv.ms/t/s!AsnjK0KGPk10gYo4_zeWH0e_Bq2FXA?e=FnLW0Y)<sup>2</sup> | [log.txt](https://1drv.ms/t/s!AsnjK0KGPk10gYo_ImRculQguFikiA?e=iLf0Wz)
|SR3D | 68.1 | - | [OneDrive](https://1drv.ms/u/s!AsnjK0KGPk10gYBcrAVJXd3w9Ckd7w?e=DWpDz8) | [68_1.txt](https://1drv.ms/t/s!AsnjK0KGPk10gYBiOCKlsxFaoQo6yA?e=BXMBgb)<sup>1</sup> / [67_6.txt](https://1drv.ms/t/s!AsnjK0KGPk10gYo8kFoHKhsMIGhWrg?e=LglnIR)<sup>2</sup> | [log.txt](https://1drv.ms/t/s!AsnjK0KGPk10gYpB05GFrJm0HIPcsg?e=SmYefu)
|NR3D | 52.1 | - | [OneDrive](https://1drv.ms/u/s!AsnjK0KGPk10gYBZFKbUir4KH37lhQ?e=FwoGCW) | [52_1.txt](https://1drv.ms/t/s!AsnjK0KGPk10gYBdNqMTotO8ai-npQ?e=lUTgka)<sup>1</sup> / [54_7.txt](https://1drv.ms/t/s!AsnjK0KGPk10gYo6J5tuU7RKTS3d-Q?e=S2GrU7)<sup>2</sup> | [log.txt](https://1drv.ms/t/s!AsnjK0KGPk10gYpASOJhMDS1ixg9QA?e=uaQCA6)

> `*`: This model is also used to evaluate the new task of **grounding without object names**, with performances of 26.5% and 21.6% for acc@0.25 and acc@0.5.    
`1`: The log of the performance we **reported in the paper**.   
`2`: The log of the performance we **retrain the model** with this open-released repository.   
> Note: To find the `overall performance`, please refer to [issue3](https://github.com/yanmin-wu/EDA/issues/3).

## 4. Training

+ Please specify the paths of `--data_root`, `--log_dir`, `--pp_checkpoint` in the `train_*.sh` script first. We use four or two 24-GB 3090 GPUs for training with a batch size of 12 by default.
+ For **ScanRefer** training
  ```
  sh scripts/train_scanrefer.sh
  ```
+ For **ScanRefer (single stage)** training
  ```
  sh scripts/train_scanrefer_single.sh
  ```
+ For **SR3D** training
  ```
  sh scripts/train_sr3d.sh
  ```
+ For **NR3D** training
  ```
  sh scripts/train_nr3d.sh
  ```

## 5. Evaluation

+ Please specify the paths of `--data_root`, `--log_dir`, `--checkpoint_path` in the `test_*.sh` script first.
+ For **ScanRefer** evaluation
  ```
  sh scripts/test_scanrefer.sh
  ```
  + **New task: grounding without object names**. Please first download our [new annotation](https://1drv.ms/u/s!AsnjK0KGPk10gYBmrVFyVts3QBpyww?e=eK2zQw), then give the path of `--wo_obj_name` in the script and run:
    ```
    sh scripts/test_scanrefer_wo_obj_name.sh
    ```
+ For **ScanRefer (single stage)** evaluation
  ```
  sh scripts/test_scanrefer_single.sh
  ```
+ For **SR3D** evaluation
  ```
  sh scripts/test_sr3d.sh
  ```
+ For **NR3D** evaluation
  ```
  sh scripts/test_nr3d.sh
  ```

## 6. Acknowledgements

We are quite grateful for [BUTD-DETR](https://github.com/nickgkan/butd_detr), [GroupFree](https://github.com/zeliu98/Group-Free-3D), [ScanRefer](https://github.com/daveredrum/ScanRefer), and [SceneGraphParser](https://github.com/vacancy/SceneGraphParser).

## 7. Citation

If you find our work useful in your research, please consider citing:
```
@inproceedings{wu2022eda,
  title={EDA: Explicit Text-Decoupling and Dense Alignment for 3D Visual Grounding},
  author={Wu, Yanmin and Cheng, Xinhua and Zhang, Renrui and Cheng, Zesen and Zhang, Jian},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

## 8. Contact

If you have any question about this project, please feel free to contact [Yanmin Wu](https://yanmin-wu.github.io/): wuyanminmax[AT]gmail.com
