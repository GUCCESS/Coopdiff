# Coopdiff (CVPR26)
CoopDiff: A Diffusion-Guided Approach for Cooperation under Corruptions
![overall](https://github.com/GUCCESS/Coopdiff/blob/main/image.png)

> [**CoopDiff: A Diffusion-Guided Approach for Cooperation under Corruptions**](https://arxiv.org/abs/2603.01688),  
> Gong Chen¹, Chaokun Zhang²*, Pengcheng Lv³  
> ¹School of Computer Science and Technology, Tianjin University  
> ²School of Cybersecurity, Tianjin University  
> ³School of Future Technology, Tianjin University  
> Accepted by CVPR 2026

### Abstract

Cooperative perception lets agents share information to expand coverage and improve scene understanding. However, in real-world scenarios, diverse and unpredictable corruptions undermine its robustness and generalization. To address these challenges, we introduce CoopDiff, a diffusion-based cooperative perception framework that mitigates corruptions via a denoising mechanism. CoopDiff adopts a teacher-student paradigm: the Quality-Aware Teacher performs voxel-level early fusion with Quality of Interest weighting and semantic guidance, then produces clean supervision features via a diffusion denoiser. The Dual-Branch Diffusion Student first separates ego and cooperative streams in encoding to reconstruct the teacher's clean targets. An Ego-Guided Cross-Attention mechanism then facilitates balanced decoding under degradation by adaptively integrating ego and cooperative features. We evaluate CoopDiff on two constructed multi-degradation benchmarks, OPV2Vn and DAIR-V2Xn, each incorporating six corruption types, including environmental and sensor-level distortions. Benefiting from the inherent denoising properties of diffusion, CoopDiff consistently outperforms prior methods across all degradation types, lowers the relative corruption error, and offers a tunable balance between precision and inference efficiency.

### Installation

```bash
# Setup conda environment
conda create -n coopdiff python=3.7 -y
conda activate coopdiff

# Install PyTorch 1.9.1 (CUDA 11.1 version)
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Install spconv (choose the correct cuda version for your system)
pip install spconv-cu113

# Install dependencies
pip install -r requirements.txt

# Build cuda extensions for bounding box NMS
python opencood/utils/setup.py build_ext --inplace

# Install OpenCOOD in development mode
python setup.py develop
```

### Quick Start

#### Train

**Step 1: Train the teacher model**

Make sure the `root_dir` in the config file (e.g. `opencood/hypes_yaml/point_pillar_diff_teacher.yaml`) points to your training dataset (e.g. `opv2v/train`).

```bash
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir ${CHECKPOINT_FOLDER}]

```

- `hypes_yaml`: path to the yaml config (e.g. teacher version)

- `model_dir` (optional): path to existing checkpoint folder for fine-tuning

**Step 2: Train the student model**

1. Copy the teacher checkpoint folder and rename it to e.g. `student_train_folder`

2. Keep only the last/best checkpoint and rename it to `epoch_1.pth`

3. Modify `core_method` in the `config.yaml` inside that folder to the student version (e.g. `point_pillar_diff_stu`)

4. Train:

```bash
python opencood/tools/train.py --model_dir student_train_folder

```

#### Test / Inference

Ensure `validation_dir` in `config.yaml` points to your test set (e.g. `opv2v/test`).

```bash
python opencood/tools/inference.py --model_dir ${student_train_folder}

```

To evaluate under different corruptions, uncomment the corresponding corruption functions (e.g. `apply_motion_blur_to_numpy`) in `opencood/data_utils/datasets/basedataset.py`.

### Citation

If you find CoopDiff helpful in your research, please consider citing:

```bibtex
@misc{chen2026coopdiffdiffusionguidedapproachcooperation,
      title={CoopDiff: A Diffusion-Guided Approach for Cooperation under Corruptions}, 
      author={Gong Chen and Chaokun Zhang and Pengcheng Lv},
      year={2026},
      eprint={2603.01688},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.01688}, 
}

```

### Acknowledgment

We sincerely thank the authors of the following high-quality open-source projects and datasets:

- [Robo3D](https://github.com/ldkong1205/Robo3D)

- [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD)

- [DSRC](https://github.com/Terry9a/DSRC/tree/main)

