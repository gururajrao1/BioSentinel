# BioSentinel — Research-Backed Model Improvements

Source: deep-research workflow on `https://github.com/gururajrao1/BioSentinel`
(12 claims verified, 15 sources). Local repo inspected 2026-11-12:
notebook-only project, 63 cells, 4 classes
(`burn`, `cuts_infection`, `necrosis`, `tissue_wound`),
ResNet-18 fine-tune + MobileNetV3-Small quantized, ~74% accuracy.

Build order: §1 → §2 → §3 → §4 → §5 (same order as below).

## 1. Fix the training recipe before changing models

- Two-stage fine-tuning: train fresh head with ImageNet backbone frozen,
  then unfreeze late stages (`layer4` + `fc`) at much lower LR.
  Prevents small-data gradients destroying pretrained features.
  - [Chest X-ray recipe](https://www.ganeshreddy.dev/xray-classification-blog.html)
  - [Two-stage ResNet-18 guide](https://offer.chtoen.com/ai/two-stage-fine-tuning-for-transfer-learning-with-resnet-18.php)
- On imbalanced ~5k-scale sets prefer inverse-frequency class weights over
  naive oversampling, conservative augmentation, discriminative LRs,
  patient-level splits. Report per-class recall, not accuracy alone.
  - [Chest X-ray recipe](https://www.ganeshreddy.dev/xray-classification-blog.html)
  - [Transfer strategies](https://visionbook.icsgen-ai.org/part-3-deep-learning-for-vision/module-21-training-recipes/section-21.3.html)
- Do not blindly add MixUp. Thyroid MIL: original-MixUp decreased accuracy,
  multilinear intra-slide helped. Validate any variant on our split.
  - [MixUp-MIL study](https://arxiv.org/html/2211.05862)
- Few-shot for rare wounds: ProtoNet meta-train mini-ImageNet/BreakHis,
  BUSI 10-shot 0.882–0.889 vs 0.831 baseline; ProtoNet > MAML, ResNet > Conv-4.
  - [Few-shot ultrasound](https://www.springerprofessional.de/few-shot-classification-of-ultrasound-breast-cancer-images-using/26996498)

BioSentinel action: freeze + head first; class weights; group splits;
conservative aug initially; log recall/F1; ProtoNet pilot for rare classes.

## 2. Data scarcity: classical first, synthetic second

- Validated wound policy (BG/D/N/P/S/V, ~75/class): TF on-the-fly
  rotation 0–30deg + brightness 0–0.2. +11% F1 over baselines
  (+10%/+11%/+7% for D/P/V low-F1 classes). Baselines ~82% VGG16 / ~84% ResNet50.
  - [Wound augmentation](https://arxiv.org/html/2411.02456v1)
- DE-GAN (+14 diabetic images): good variation, no proven F1 gain.
- DermSynth3D (3D-mesh blending, multi-view/lighting): real+synthetic improves
  detection, converges ~400 synthetic; synthetic-only Dice 0.49 vs 0.81 real;
  synthetic pretraining beats COCO at 10/50/100/610 real.
  - [DermSynth3D](https://ar5iv.labs.arxiv.org/abs/2305.12621)

BioSentinel action: rotation/brightness baseline first; cap synthetic ~400;
pretrain-then-finetune; require held-out gain for any GAN/diffusion images.

## 3. Architecture upgrades beyond ResNet-18

Shared AZH test (n=234): VGG-19 78.63%/78.14/0.956; EfficientNet-B4
83.76%/83.36/0.975; DINOv2+Linear 86.75%/87.01/0.986; WILLIE up to 91.88%.
  - [WILLIE](https://arxiv.org/html/2610.05341v1)

- DINOv2+ConvNeXt suffices for classification; SAM2 adds dense gains.
  Segment-then-derive-box beats YOLOv8m/RT-DETR-L on mask-derived GT.
- Edge: YOLO11n 6.4 GFLOPs F1 0.836/mAP 0.893 vs VGG+MLP 82.48%,
  EfficientNet 83.19%.
  - [YOLO11n wound](https://www.techscience.com/cmc/v85n1/63526/html)
- Cross-dataset: EfficientNet-B4 92.3%, ViT-B/16 95.8%,
  ResNet-50+DINOv2+Swin-T ensemble 99.90%.
  - [WoundNet-Ensemble](https://arxiv.org/html/2512.18528v1)
- Multimodal: ViT+Wavelet 0.8401, +location Transformer 0.8644.
  - [ViT+Wavelet](https://arxiv.org/html/2504.10452v1)
- Segmentation: ERYXSeg 0.7633 IoU/0.8658 Dice foot-ulcer, 0.6910/0.8173 mixed;
  FUSeg winner UNet-EfficientNetB2 + LinkNet-EfficientNetB1 88.80% Dice.
  - [ERYXSeg](https://www.frontiersin.org/journals/surgery/article/10.3389/fsurg.2026.1790725/full)

BioSentinel action: EfficientNet-B4 or YOLO-nano for edge;
DINOv2/ViT or ensemble for max accuracy; EfficientNet-UNet or ERYXSeg hybrid;
derive boxes from masks; add wavelet/location if available.

## 4. Evaluate like a clinical model, not a leaderboard

- Bootstrap/repeated CV for optimism; external = different site/time/device;
  temporal tests drift.
- Report AUROC + calibration (plot/slope/intercept) + usefulness together.
- Follow TRIPOD+AI (27 items, supersedes TRIPOD 2015), PROBAST+AI, CLAIM/STARD-AI.
  - [Palo Alto validation guide](https://www.paloaltohealthlabs.com/resources/clinical-prediction-model-validation.html)
  - [Frontiers review](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1890072/full)
- Exemplar: patient-stratified 80/20 holdout (4988/1247) + 5-fold CV inside
  train (3990/998); report accuracy/sensitivity/specificity/AUC, ROC/PR.
  RepGhost 81.4%, AUC 0.894±0.012.
  - [Hypergranulation study](http://pmc.ncbi.nlm.nih.gov/articles/PMC12790267)
- Reduce partition bias: identical aug/hparams, no sample in train+val,
  plus OOD wound dataset.
  - [WoundAmbit](https://extrapolator.ai/2026/10/12/woundambit-benchmarks-segmentation-models-for-chronic-wound-tracking/)
- Prevent leakage: keep repeat encounters/patients out of both splits.
  - [Prediction Toolkit](https://blog.dataindeed.org/series/toolkit/posts/07-prediction_modeling_toolkit.html)
- Expect drops under shift: hundreds of COVID models, none clinically useful.

BioSentinel action: lock patient holdout; nested 5-fold CV; AUROC + calibration
+ sensitivity/specificity; OOD/device-shift test; TRIPOD+AI checklist.

## 5. Battlefield / edge deployment

- ENet/TopFormer/UNeXt match large UNet (IoU 69–72%, DSC 82–84%),
  TorchScript live at 224×224; TopFormer best vs wound-coloured objects.
  - [Lightweight mobile](https://ar5iv.labs.arxiv.org/html/2407.07605)
- Structured pruning 40% channels ~1.5x faster; fine-tune after or prune gradually.
- INT8 ~4x mem, 2–3x speed, typically <1% loss; distill→prune→quantize chains.
  - [Edge AI guide](https://wricktalukdar.technology/articles/edge-ai-deploying-models-resource-constrained.html)
- Jetson Nano: PTQ 3.8x shrink, up to 3x latency cut, cost up to ~3.2 mAP;
  QAT recovers (YOLOv5s within 0.9 mAP).
  - [Jetson quantization](https://iieta.org/journals/isi/paper/10.18280/isi.310530)
- Practice: PTQ + calibration 80–95% FP32 at 2–4x speedup; QAT for medical;
  TensorRT on INT8 ONNX adds 1.5–3x.
  - [CV in Production](https://www.mlaia.com/blog-computer-vision-production.html)

BioSentinel action: ship TopFormer/ENet or YOLO-nano TorchScript/ONNX-INT8 for
offline field use; calibrate PTQ on wound images; QAT for clinical model;
measure FPS, memory, per-class recall drop on-device.

## Caveats

- No BioSentinel code was available to the research agents (empty workspace);
  mapping above was added after inspecting the notebook on 2026-11-12.
- WoundNet 99.90% is dataset-specific, not head-to-head with AZH. WILLIE
  localization uses mask-derived boxes. YOLO11n mAP vs accuracy mixes metrics.
- No battlefield-wound quantization study found; edge numbers transfer from
  foot-ulcer mobile + COCO/Jetson detection.
- Open: rarest classes, body-location availability, edge target (phone vs
  Jetson), OOD site/device.


