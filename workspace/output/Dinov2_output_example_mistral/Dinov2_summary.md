# DINOv2: Learning Robust Visual Features without Supervision 

**Authors:**
- Maxime Oquab, Timothée Darcet, Théo Moutakanni
- Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza
- Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba
- Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu
- Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski

**Affiliation:**
- Meta AI Research
- Inria

**Notes:**
- * indicates equal contribution
- ${ }^{1}$ indicates Inria affiliation

#### Abstract

**Detailed Summary:**

- **Breakthroughs in Computer Vision:**
  - **Foundation Models:** Similar to NLP, foundation models can simplify image use in systems.
  - **General-Purpose Visual Features:** These features work across image distributions and tasks without fine-tuning.

- **Pretraining Methods:**
  - **Self-Supervised Methods:** Existing pretraining methods, especially self-supervised, can produce general-purpose features.
  - **Curated Data:** Training on diverse, curated data from various sources is crucial.

- **Technical Contributions:**
  - **Data Pipeline:** Proposed an automatic pipeline to build a dedicated, diverse, and curated image dataset.
  - **Model Size:** Trained a ViT model with 1B parameters and distilled it into smaller models.
  - **Performance:** Smaller models surpassed OpenCLIP (Ilharco et al., 2021) on most image and pixel-level benchmarks.

- **!\[\[Image\]\]**

## 1 Introduction

**Detailed Summary:**

- **Shift in NLP:** Learning task-agnostic pretrained representations has become standard in NLP, with models achieving high performance on downstream tasks without fine-tuning.
- **Paradigm Shift Expected in Computer Vision:** Similar "foundation" models are expected to appear in computer vision, generating visual features that work out-of-the-box on any task.
- **Text-Guided Pretraining:** Most promising efforts focus on text-guided pretraining, but this limits the information retained about the image and requires aligned text-image corpora.
- **Self-Supervised Learning:** An alternative is self-supervised learning, which can capture information at the image and pixel level and has enabled various applications. However, most advances were made on small curated datasets like ImageNet-1k.
- **Current Work:** This work explores if self-supervised learning can learn general-purpose visual features when pretrained on a large quantity of curated data.
- **Technical Contributions:** The work revisits existing discriminative self-supervised approaches and makes improvements to stabilize and accelerate learning, making the approach faster and more memory-efficient.
- **Pretraining Data:** An automatic pipeline is used to filter and rebalance datasets from an extensive collection of uncurated images, inspired by pipelines used in NLP.
- **Pretrained Models:** The work provides a variety of pretrained visual models, called DINOv2, trained with different Vision Transformers (ViT) architectures on the curated data.
- **Validation:** DINOv2 is validated on various computer vision benchmarks at both image and pixel levels, showing that self-supervised pretraining alone can learn transferable frozen features competitive with the best openly available weakly-supervised models.

# 2 Related Work 

**Self-Supervised Learning Approaches:**

- **Intra-Image Methods:**
  - **Pretext Tasks:**
    - Context prediction (Doersch et al., 2015)
    - Re-colorization (Zhang et al., 2016)
    - Transformation prediction (Gidaris et al., 2018)
    - Inpainting (Pathak et al., 2016)
    - Patch re-ordering (Noroozi & Favaro, 2016; Misra & Maaten, 2020)
  - **Masked Autoencoders (MAE):**
    - He et al. (2022): MAE learns features that improve downstream tasks with finetuning.
    - Validated on video (Tong et al., 2022), audio (Xu et al., 2022), and other modalities (Girdhar et al., 2023).

- **Discriminative Self-Supervised Learning:**
  - **Instance Classification:**
    - Dosovitskiy et al. (2016)
    - Bojanowski & Joulin (2017)
    - Wu et al. (2018)
  - **Improvements:**
    - Instance-level objectives (Hénaff et al., 2019; He et al., 2020; Chen & He, 2021; Chen et al., 2020; Grill et al., 2020; Caron et al., 2021)
    - Clustering (Caron et al., 2018; Asano et al., 2020; Caron et al., 2020)
  - **Scaling Self-Supervised Pretraining:**
    - Caron et al. (2019)
    - Goyal et al. (2019, 2022a)
    - Tian et al. (2021)
    - Goyal et al. (2021): Benefits from scaling in model size with enough pretrained data.

**Data Curation:**

- **Inspiration:** Text curation pipelines (Wenzek et al., 2020)
- **Approach:** Uses visual similarity between images for filtering uncurated datasets without metadata, supervision, or pretrained encoders.
- **Pipeline Overview:** ![img_3.jpeg](images\img_3.jpeg)
  - Images from curated and uncurated sources are mapped to embeddings.
  - Uncurated images are deduplicated and matched to curated images.
  - The resulting combination augments the initial dataset through a self-supervised retrieval system.

# 3 Data Processing 

**LVD-142M Dataset Assembly Pipeline**

- **Data Sources**
  - **Curated Datasets**
    - ImageNet-22k
    - ImageNet-1k train split
    - Google Landmarks
    - Fine-grained datasets (detailed in appendix)
  - **Uncurated Data Source**
    - 1.2B unique images from crawled web data
    - Extracted from <img> tags, post-processed (PCA hash deduplication, NSFW filtering, face blurring)

- **Deduplication**
  - Copy detection pipeline (Pizzi et al., 2022)
  - Removed near-duplicates from uncurated data and benchmark test/validation sets

- **Self-Supervised Image Retrieval**
  - Image embedding using ViT-H/16 pretrained on ImageNet-22k
  - Cosine-similarity for distance measure
  - K-means clustering of uncurated data
  - Retrieval: N=4 nearest neighbors or M sampled images from corresponding cluster

- **Implementation Details**
  - Faiss library for efficient indexing and batch searches
  - GPU-accelerated indices with inverted file and product quantization codes
  - Processing distributed on 20 nodes with 8 V100-32GB GPUs, completed in less than two days

# 4 Discriminative Self-supervised Pre-training 

**Detailed Summary:**

- **Self-Supervised Learning Method:**
  - Combines DINO, iBOT, and SwAV losses.
  - Adds KoLeo regularizer and high-resolution training phase.

- **Image-Level Objective (DINO):**
  - Cross-entropy loss between student and teacher features.
  - Student: DINO head on class token, softmax for $p_s$.
  - Teacher: DINO head on class token, centering (moving avg. or Sinkhorn-Knopp) for $p_t$.
  - Loss: $\mathcal{L}_{DINO} = -\sum p_t \log p_s$.

- **Patch-Level Objective (iBOT):**
  - Student: iBOT head on masked patch tokens, softmax for $p_s$.
  - Teacher: iBOT head on visible patch tokens, centering for $p_t$.
  - Loss: $\mathcal{L}_{iBOT} = -\sum_i p_ti \log p_{si}$.

- **Head Weight Untying:**
  - Separate heads for DINO and iBOT losses.

- **Sinkhorn-Knopp Centering:**
  - Replaces softmax-centering in DINO and iBOT.
  - 3 iterations of Sinkhorn-Knopp algorithm.

- **KoLeo Regularizer:**
  - Encourages uniform feature span within a batch.
  - Loss: $\mathcal{L}_{koleo} = -\frac{1}{n} \sum_i \log(d_{n,i})$.

- **Adapting Resolution:**
  - High-resolution training phase at end of pretraining.
  - Resolution: $518 \times 518$.

# 5 Efficient implementation 

**Detailed Summary:**

- **Model Training Improvements:**
  - Trained models on A100 GPUs using PyTorch 2.0.
  - Code and pretrained models available under Apache 2.0 license.
  - DINOv2 runs 2x faster and uses 1/3 of the memory compared to iBOT on the same hardware.

- **Fast and Memory-Efficient Attention:**
  - Implemented FlashAttention for improved memory usage and speed in self-attention layers.
  - ViT-g architecture uses 1536 embedding dimension with 24 heads for optimal compute efficiency.
  - No significant differences in final accuracy compared to the original architecture.

- **Sequence Packing:**
  - Concatenates sequences of different lengths into a single long sequence for transformer blocks.
  - Block-diagonal mask prevents attention between different sequences.
  - xFormers library provides lower-level components for this setup.

- **Efficient Stochastic Depth:**
  - Improved version skips computation of dropped residuals instead of masking the result.
  - Randomly shuffles samples over the batch dimension and slices the first (1-d) * B samples for computations.

- **Fully-Sharded Data Parallel (FSDP):**
  - Reduces memory footprint per GPU by splitting model replicas across GPUs.
  - PyTorch-FSDP mixed-precision reduces communication costs by 50% compared to DDP.
  - Scales more efficiently than DDP with float16 autocast when scaling the number of GPU nodes.

- **Model Distillation:**
  - Distills smaller models from the largest model (ViT-g) instead of training from scratch.
  - Leverages knowledge distillation to reproduce the output of a large model with a smaller model.
  - Uses a larger model as a frozen teacher and keeps a spare EMA of the student.
  - Achieves better performance than training from scratch, even for a ViT-L.

# 6 Ablation Studies 

- **Pipeline Ablations**:
  - **Technical Modifications**:
    - Ablation 1: Remove modification A
    - Ablation 2: Remove modification B
    - Ablation 3: Remove modification C
  - **Pretraining Data**:
    - Ablation 4: Use only dataset X
    - Ablation 5: Use only dataset Y
    - Ablation 6: Use only dataset Z
  - **Model Distillation**:
    - Ablation 7: No distillation
    - Ablation 8: Distill from model A
    - Ablation 9: Distill from model B
- **Downstream Tasks**:
  - **Task 1**: Description and results
  - **Task 2**: Description and results
  - **Task 3**: Description and results

### 6.1 Improved Training Recipe

**Summary:**

- **Model Improvements over iBOT:**
  - **Baseline iBOT:**
    - INet-1k k-NN: 72.9
    - INet-1k linear: 82.3
  - **Successive Component Additions:**
    - +Reproduction: INet-1k k-NN: 74.5, INet-1k linear: 83.2
    - +LayerScale, Stochastic Depth: INet-1k k-NN: 75.4, INet-1k linear: 82.0
    - +128k prototypes: INet-1k k-NN: 76.6, INet-1k linear: 81.9
    - +KoLeo: INet-1k k-NN: 78.9, INet-1k linear: 82.5
    - +SwiGLU FFN: INet-1k k-NN: 78.7, INet-1k linear: 83.1
    - +Patch size 14: INet-1k k-NN: 78.9, INet-1k linear: 83.5
    - +Teacher momentum 0.994: INet-1k k-NN: 79.4, INet-1k linear: 83.6
    - +Tweak warmup schedules: INet-1k k-NN: 80.5, INet-1k linear: 83.8
    - +Batch size 3k: INet-1k k-NN: 81.7, INet-1k linear: 84.7
    - +Sinkhorn-Knopp: INet-1k k-NN: 81.7, INet-1k linear: 84.7
    - +Untying heads = DINOv2: INet-1k k-NN: 82.0, INet-1k linear: 84.5

- **Pretraining Data Ablation:**
  - INet-22k: INet-1k: 85.9, Im-A: 73.5, ADE-20k: 46.6, Oxford-M: 62.5, iNat2018: 81.1, iNat2021: 85.6, Places205: 67.0
  - INet-22k \ INet-1k: INet-1k: 85.3, Im-A: 70.3, ADE-20k: 46.2, Oxford-M: 58.7, iNat2018: 80.1, iNat2021: 85.1, Places205: 66.5
  - Uncurated data: INet-1k: 83.3, Im-A: 59.4, ADE-20k: 48.5, Oxford-M: 54.3, iNat2018: 68.0, iNat2021: 76.4, Places205: 67.2
  - LVD-142M: INet-1k: 85.8, Im-A: 73.9, ADE-20k: 47.7, Oxford-M: 64.6, iNat2018: 82.3, iNat2021: 86.4, Places205: 67.6

# 6.2 Pretraining Data Source 

**Summary:**

- **Pretraining Dataset Impact:**
  - Curated data outperforms uncurated data in self-supervised pretraining.
  - LVD-142M outperforms ImageNet-22k on most benchmarks, except ImageNet-1k.
  - Diverse and large-scale data improves performance on unseen domains.

- **Model Scale vs Data Scale (Figure 4):**
  - ViT-g trained on LVD-142M outperforms ViT-g trained on ImageNet-22k on most benchmarks.

- **Ablation Study (Table 3):**
  - KoLeo loss term improves nearest-neighbor search tasks (e.g., retrieval).
  - Masked Image Modeling (MIM) loss term improves patch-level tasks (e.g., segmentation).
  - LVD-142M offers a balanced dataset for overall best performance.

# 6.3 Model Size and Data 

- **Model Size vs. Data Size Impact**:
  - Larger models benefit more from larger datasets like LVD-142M compared to smaller datasets like ImageNet-22k.
  - **ViT-g Performance Comparison**:
    - Trained on LVD-142M, ViT-g matches ImageNet-1k performance of a model trained on ImageNet-22k.
    - Outperforms ImageNet-22k trained model on other benchmarks.

### 6.4 Loss Components

**Impact of Loss Term Ablation:**

- **KoLeo Loss:**
  - **Instance Retrieval:** Improved by over 8% on Oxford-M.
  - **ImageNet-1k & ADE-20k:** No significant performance loss.

- **Masked Image Modeling Term (iBOT):**
  - **ADE-20k:** Improved by nearly 3%.
  - **ImageNet-1k & Oxford-M:** No significant performance loss.

### 6.5 Impact of Knowledge Distillation

**Detailed Summary:**

- **Knowledge Distillation:**
  - **ViT-L/14 vs ViT-g/14:**
    - Trained from scratch: ViT-L/14 underperforms ViT-g/14 on all 12 benchmarks.
    - Distilled from ViT-g/14: ViT-L/14 outperforms ViT-g/14 on all benchmarks.
  - **Averaged metrics on 8 vision tasks:** Distilled ViT-L/14 outperforms both ViT-L/14 and ViT-g/14.
  - ![img_5.jpeg](images\img_5.jpeg)
  - ![img_6.jpeg](images\img_6.jpeg)

- **Role of Resolution:**
  - **ViT-L/16 on ImageNet-1k:**
    - Training at high resolution (416) for a short duration outperforms training at low resolution (224) for the full duration.
    - Training at low resolution (224) then high resolution (416) for a short duration achieves results close to full high-resolution training.

# 6.6 Impact of Resolution 

**Summary:**

- **Impact of Resolution Change during Pretraining:**
  - **Training at Fixed Resolution:**
    - $224 \times 224$: Standard training resolution.
    - $416 \times 416$: Compute-intensive, high-resolution training.
  - **Mixed Resolution Training:**
    - Initial training at $224 \times 224$, resume at $416 \times 416$ for 10k iterations.
  - **Ablation Study:**
    - ViT-L/16 trained on ImageNet1k.
    - Linear probe performance on ImageNet-1k and ADE20k at various resolutions.
  - **Findings:**
    - High-resolution training improves performance across resolutions.
    - Training at $416 \times 416$ is approximately $3 \times$ more compute-intensive than $224 \times 224$.
    - Mixed resolution training (10k iterations at $416 \times 416$) performs nearly as well with significantly less compute.
  - **Conclusion:**
    - Mixed resolution training is more efficient, included at the end of pretraining.

# 7 Results 

**Empirical Evaluation of Our Models on Image Understanding Tasks**

- **Purpose**:
  - Demonstrate superior performance of self-supervised features over current state-of-the-art.
  - Show competitive performance against weakly-supervised models on various tasks.

- **Evaluated Tasks**:
  - Global and local image representations
  - Category and instance-level recognition
  - Semantic segmentation
  - Monocular depth prediction
  - Action recognition

- **Benchmarks**:
  - Detailed list in Appendix C

- **Baselines**:
  - **Self-supervised models**:
    - MAE (He et al., 2022)
    - DINO (Caron et al., 2021)
    - SEERv2 (Goyal et al., 2022a)
    - MSN (Assran et al., 2022)
    - EsViT (Li et al., 2022a)
    - Mugs (Zhou et al., 2022b)
    - iBOT (Zhou et al., 2022a)
  - **Weakly-supervised models**:
    - CLIP (Radford et al., 2021)
    - OpenCLIP (Ilharco et al., 2021; Cherti et al., 2023)
    - SWAG (Singh et al., 2022)
    - OpenCLIP-G (best performing)

- **Evaluation Details**:
  - ImageNet-1k: Reported performance for all mentioned methods.
  - Other evaluations: Reported top four best-performing self-supervised models.

### 7.1 ImageNet Classification

**Detailed Summary:**

- **ImageNet-1k Classification:**
  - Trained a simple classifier on frozen backbone, no finetuning.
  - Evaluated on ImageNet-ReaL and ImageNet-V2.
  - Compared to previous state-of-the-art (iBOT ViT-L/16), showed significant improvement (+4.2%) on linear evaluation.
  - Better performance on alternative test sets, indicating stronger generalization.

- **Weakly-Supervised Models Comparison:**
  - Compared to open-source weakly-supervised models (OpenCLIP, EVA-CLIP) on ImageNet-1k.
  - Surpassed OpenCLIP with ViT-G/14 (+0.3%) and EVA-CLIP with ViT-g/14 (+0.1%).
  - Better performance on ImageNet-V2 (+1.1% vs EVA-CLIP), indicating better generalization.

- **Finetuning the Encoders:**
  - Finetuned models on ImageNet-1k using Touvron et al. (2022) pipeline.
  - Top-1 accuracy improved by more than +2% for both 224 and 448 resolutions.
  - Best finetuned performance (88.9%) was -2.2% from the absolute state-of-the-art.

- **Robustness Analysis:**
  - Evaluated on domain generalization benchmarks (ImageNet-A, ImageNet-R, ImageNet-C, Sketch).
  - Dramatically better robustness compared to iBOT (+29.6% on A, +22.1% on R, +23.0% on Sketch).
  - Improved upon best weakly-supervised model on ImageNet-A, but lagged behind on R and Sketch.

- **Other Image and Video Classification:**
  - Evaluated on iNaturalist 2018/2021, Places205, Kinetics-400, UCF-101, and SSv2.
  - Best performance on iNaturalist 2018/2021 and Kinetics-400, competitive on other benchmarks.

# 7.2 Additional Image and Video classification Benchmarks 

**Summary:**

- **Downstream Classification Benchmarks:**
  - **Large & Fine-grained Datasets:**
    - iNaturalist 2018, 2021: +8.6%, +9.7% over OpenCLIP ViT-G/14
    - Places205: -2.3% compared to OpenCLIP ViT-G/14
  - **SimCLR Tasks:**
    - UCF101, Kinetics-400: Matches OpenCLIP's performance
    - Something-Something v2: +2.5% over OpenCLIP

- **Video Action Recognition:**
  - UCF101, Kinetics-400: Matches OpenCLIP's performance
  - Something-Something v2: +2.5% over OpenCLIP

- **Fine-grained Benchmarks (Table 8):**
  - Average accuracy: 92.1% (ViT-g/14)

- **Instance-level Recognition (Table 9):**
  - Average accuracy: 76.5% (ViT-g/14)

- **CUB Dataset:**
  - Outperforms state-of-the-art SSL models on Stanford Cars and FGVC Aircraft
  - Competitive with OpenCLIP on most classification benchmarks, except SUN and Cars

# 7.3 Instance Recognition 

**Detailed Summary:**

- **Instance-Level Recognition:**
  - **Method:** Non-parametric approach using cosine similarity ranking of images from a database.
  - **Datasets:** Paris, Oxford (landmark recognition benchmarks), Met (artworks from the Metropolitan Museum), AmsterTime (street view images matched to archival images of Amsterdam).
  - **Evaluation:**
    - Mean Average Precision (mAP) used as performance metric.
    - Results in Table 9 show significant improvement over SSL (+41% mAP on Oxford-Hard) and weakly-supervised (+34% mAP on Oxford-Hard) features.
    - Features perform well across task granularities, both at category-level and instance-level.

- **Feature Extraction Models (Table 10):**
  - **Models:** OpenCLIP, MAE, DINO, iBOT, DINOv2, ViT-S/14, ViT-B/14, ViT-L/14, ViT-g/14.
  - **Evaluation:** Semantic segmentation on ADE20K, CityScapes, and Pascal VOC with frozen features and a linear classifier, and with multiscale.
  - **Results:** ViT-g/14 shows the best performance across all datasets, with and without multiscale. Using a ViT-Adapter on top of the frozen ViT-g/14 backbone gives 60.2 mIoU on ADE-20k.

# 7.4 Dense Recognition Tasks 

**Detailed Summary:**

- **Patch-Level Feature Quality Probing:**
  - **Semantic Image Segmentation:**
    - **Evaluation Setups:**
      - Linear: Simple linear layer for prediction, low-resolution logit map upsampled.
      - +ms: Boosted linear setup with larger resolution, multiscale test-time augmentations.
    - **Results:**
      - Excellent performance on all datasets and setups.
      - +ms setup matches finetuned MAE with Upernet decoder (53.0 vs 53.6 mIoU).
      - Best model nearly matches state-of-the-art on Pascal VOC (86.2 vs 89.0 mIoU).
  - **Frozen Backbone in SOTA Pipeline:**
    - **Setup:** ViT-Adapter with Mask2former head, 66% of weights frozen.
    - **Results:** 60.2 mIoU on ADE20k, close to competitive state-of-the-art (62.9 mIoU).
  - **Monocular Depth Estimation:**
    - **Evaluation Setups:**
      - lin. 1: Single layer with [CLS] token, bilinear upsampling, linear layer with classification loss.
      - lin. 4: Four layers with [CLS] token, bilinear upsampling, linear layer with classification loss.
      - DPT: DPT decoder with regression task.
    - **Results:**
      - Features surpass best SSL and WSL features available.
      - iBOT features from ViT-L outperform OpenCLIP with ViT-G.
      - Model with DPT decoder matches or exceeds Li et al. (2022b).
      - Good out-of-domain generalization on SUN-RGBD.

# 7.5 Qualitative Results 

**Detailed Summary:**

- **Qualitative Analyses:**
  - **Dense Prediction Evaluations:**
    - **Semantic Segmentation & Depth Estimation:**
      - DINOv2 vs OpenCLIP with linear classifier on ADE20K, NYUd, KITTI, SUN RGB-D.
      - DINOv2 produces better segmentation masks with fewer artifacts.
      - DINOv2 estimates depth more smoothly with less artifacts.
      - OpenCLIP ignores objects like chairs in SUN RGB-D.
  - **Out-of-Distribution Generalization:**
    - Depth prediction and segmentation linear classifiers applied to out-of-distribution examples.
    - DINOv2 features transfer well between domains (animals, paintings).
  - **PCA of Patch Features:**
    - Separates main object from background using first PCA component.
    - Other components correspond to 'parts' of objects, matching well for same category images.
  - **Patch Matching:**
    - Matches semantic regions across images (e.g., wing of plane to wing of bird).
    - Robust to style and pose variations (e.g., elephant).

- **Figures:**
  - Figure 7: Segmentation and depth estimation examples from ADE20K, NYUd, SUN RGB-D, KITTI.
  - Figure 8: Out-of-distribution examples with frozen DINOv2-g features and a linear probe.
  - Figure 9: More PCA components visualization, matching parts between related images.
  - Figure 10: Patch matching examples across images.

# 8 Fairness and Bias Analysis 

- **Fairness Evaluations of Models**
  - **Geographical Fairness**
    - Largest ViT-g model used
    - No significant geographical biases found
  - **Harmful Label Associations**
    - Largest ViT-g model used
    - No potential harmful label associations identified

### 8.1 Geographical Fairness

**Dollar Street Dataset Evaluation:**

- **Dataset Details:**
  - Introduced by De Vries et al. (2019)
  - 16,073 images from 289 households across 54 countries
  - Task: Recognize 94 visual concepts varying by income or location

- **Evaluation Protocol (Goyal et al., 2022b):**
  - Compares performance across countries and income levels

- **Model Comparison (Table 12):**
  - Our model (DINOv2) vs SEERv2 (Goyal et al., 2022a)
  - DINOv2 slightly fairer across regions and incomes than SEERv2
  - DINOv2 significantly better than supervised baseline (Goyal et al., 2022a)

- **Regional Performance:**
  - DINOv2 drops by 25.7% in Africa compared to Europe
  - Biased toward Western countries

- **Income-based Performance:**
  - DINOv2 performs 31.7% better on high-income households than low-income ones

- **Model Limitations:**
  - Significant biases toward wealthy households from Western countries

- **Matching Across Images (Figure 10):**
  - Matches patch-level features between images from different domains, poses, and objects with similar semantics
  - Exhibits ability to transfer across domains and understand relations between similar parts of different objects

# 8.2 Gender, Skintones and Age 

**Summary:**

- **Evaluation of Model's Fairness:**
  - **Gender, Skin Tone, Age Group Analysis:**
    - Model classifies images of all groups predominantly as 'Human'.
    - No significant deviations across skin tones.
    - Neither SEERv2 nor DINOv2 predict harmful labels (Non-Human, Crime).
    - Model often predicts 'Possibly-Human' class, especially for men due to 'Beard' class prevalence.
    - No clear bias against any particular group observed.

- **Carbon Footprint of Reproducing DINOv2:**
  - **GPU Type:** A100-40GB
  - **GPU Power Consumption:** 400W
  - **GPU-hours:** 22,016
  - **PUE:** 1.1
  - **Total Power Consumption:** 9.7 MWh
  - **Carbon Emitted:** 3.7 tCO₂

# 9 Estimating the Environmental Impact of Training our Models 

**Carbon Emission Estimation for Model Training**

- **Methodology**
  - Patterson et al. (2021) propose a complex methodology to estimate carbon emissions based on data center specifics.
  - Alternative methodology: Estimate retraining a similar model in an average US data center.
    - Fixed variables: PUE (1.1), carbon intensity factor (0.385 kg CO2 eq/KWh).
    - Power consumption of A100-80GB: 400 W.

- **Retraining DINOv2 ViT-g**
  - Energy consumption: 2.24 MWh.
  - Carbon emission: 837.6 kg CO2 eq.
  - Comparison: OpenCLIP ViT-L (22.4 MWh, 8.48 t CO2 eq), OpenCLIP ViT-G (118.9 MWh, 44.5 t CO2 eq).

- **Whole Project Carbon Footprint**
  - Estimated range: 0.5 k - 1 k tCO2 eq.
  - Equivalent to 200 k GPU-days.
  - Primary sources: Self-supervised pre-trainings.
    - ViT-g pre-training (22 k GPU-hours): 3.7 t CO2 eq.
    - ImageNet-1k finetuning (1 k GPU-hours): 0.2 t CO2 eq.
  - Excludes manufacturing and disposal emissions.

# 10 Future work and Discussion 

**DINOv2: A New Series of Image Encoders**

- **Pretraining and Performance:**
  - First self-supervised learning (SSL) work on images to match weakly supervised alternatives.
  - No finetuning required to achieve high performance across various benchmarks.

- **Factors Contributing to Strong Performance:**
  - Improved training recipe with better hyperparameters and regularization (Table 1).
  - Larger model scale with improved results (Fig. 4).
  - Larger dataset (Fig. 4).
  - Distillation process benefiting smaller models (Fig. 5).

- **Emerging Properties:**
  - Understanding of object parts and scene geometry, regardless of image domains.
  - Expectation of more properties to emerge at larger scales, similar to instruction emergence in large language models.

- **Compatibility and Future Work:**
  - Visual features compatible with simple linear classifiers.
  - Plan to leverage this ability to train a language-enabled AI system processing visual features as word tokens for information extraction.

## Acknowledgments.

- **Acknowledgments**:
  - **Mathilde Caron**: Initial discussions.
  - **Julien Mairal**:
    - ERC grant: APHELAIA project (101087696).
    - ANR 3IA: MIAI@Grenoble Alpes (ANR-19-P3IA-0003).
  - **Olivia Joulin**: Horse drawing in Fig. 10.
  - **Madeleine and Léon**: Posing for Fig. 8.
  - **FAIR and Meta AI**: Feedback throughout the project.

# A Data Processing 



## A. 1 Data selection

**Datasets for LVD-142M:**

- **Image-Level Recognition:**
  - **COCO:**
    - 330K images, 1.5 million object instances.
    - 80 object categories, 91 thing and 50 stuff classes.
  - **ImageNet:**
    - 1.2 million images, 1000 classes.
    - Broad range of object categories.
  - **VOC2012:**
    - 11,540 images, 20 object categories.
    - Dense annotations for object detection and segmentation.

- **Dense Recognition:**
  - **Cityscapes:**
    - 5000 images, 19 semantic classes.
    - High-resolution images, detailed annotations.
  - **KITTI:**
    - 7481 images, 3D object detection and semantic segmentation.
    - Real-world driving scenarios, diverse weather conditions.
  - **Mapillary Vistas:**
    - 600K images, 66 semantic classes.
    - Large-scale, diverse street-level imagery.

## A. 2 Image similarity

- **Image Similarity Comparison**:
  - **Formula**: $m(s, r) = \text{cosine-similarity}(f(s), f(r)) = \frac{f(s) \cdot f(r)}{\|f(s)\|_{2}\|f(r)\|_{2}}$
  - **Application**: Used to compare image features, either from our model or generated for deduplication.

## A. 3 Deduplication

**Deduplication Process:**

- **Self-deduplication:**
  - Use Pizzi et al. (2022) embeddings.
  - Retrieve 64 nearest neighbors (cosine similarity > 0.6).
  - Extract connected components using disjoint set data structure.
  - Keep one representative per duplicate component.
  - Result: 1.1B images.

- **Relative deduplication:**
  - Discard images similar to train and test splits (similarity > 0.45).
  - Identify and remove duplicate components.
  - Result: 744M images.

## A. 4 Retrieval

**Dataset Augmentation via Retrieval**

- **Approaches**
  - **Sample-based**
    - Suitable for datasets larger than 1M images
    - Collects a fixed number of nearest images for each sample image
    - Multiplies dataset size by the fixed number
    - Used with $k=4$ for Google Landmarks v2 and ImageNet-22k
    - Used with $k=32$ for LVD-142M dataset
  - **Cluster-based**
    - Suitable for smaller datasets
    - Clusters uncurated data source into 100,000 separate clusters using distributed $k$-means
    - Picks 10,000 images from each cluster with more than 3 images in the retrieved dataset
    - Limits retrieval to a maximum of 1M images per dataset to maintain balance

## B Implementation Details



## B. 1 Unsupervised pre-training

**Summary:**

- **Pre-training Datasets:**
  - **ImageNet-22k:**
    - Classification: 14,197,086 images (as is), 56,788,344 images (sampled)
    - Fine-grained classification: 14,197,086 images (sampled)
  - **ImageNet-1k:**
    - Classification: 1,281,167 images (sampled)
  - **Fine-grained classification datasets:**
    - Caltech 101, CUB-200-2011, DTD, FGVC-Aircraft, Flowers-102, Food-101, Oxford-IIIT Pet, Stanford Cars, SUN397, Pascal VOC 2007: 1,000,000 images each (clustered)
  - **Segmentation datasets:**
    - ADE20K, Cityscapes, Pascal VOC 2012 (seg.): 1,000,000 images each (clustered)
  - **Depth estimation datasets:**
    - Mapillary SLS, KITTI, NYU Depth V2, SUN RGB-D: 1,000,000 images each (clustered)
  - **Retrieval datasets:**
    - Google Landmarks v2, AmsterTime, Met, Revisiting Oxford, Revisiting Paris: 1,000,000 images each (clustered)
  - **Total:** 142,109,386 images

- **Pre-training Details:**
  - **Models:** DINOv2-S, DINOv2-B, DINOv2-L (distilled and from scratch), DINOv2-g
  - **Architectures:** ViT-S/14, ViT-B/14, ViT-L/14, ViT-g/14
  - **Hyperparameters:** Dropout rates, learning rates, batch sizes, optimizer, learning rate schedule, weight decay schedule, teacher momentum schedule, precision
  - **Regularization:** KoLeo regularizer with weight 0.1 between class tokens of the first global crop
  - **EMA update:** Teacher network is an exponential moving average of the student network with a momentum value in [0.994, 1.0] following a cosine schedule

# B. 2 High-Resolution adaptation 

- **Model Initialization and Fine-tuning:**
  - Load pretrained weights
  - Train for 10,000 iterations
  - Compress schedules to fit 10,000 iterations
  - Keep all hyperparameters constant, except:
    - Reduce base learning rate

## B. 3 Linear probing evaluation

**Linear Probing Evaluation Parameters and Grid Search:**

- **Evaluation Parameters:**
  - Learning rate: $\{0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5\}$
  - Output layers: $\{1, 4\}$
  - Concatenate average-pooled tokens: $\{yes, no\}$

- **Training:**
  - Optimizer: SGD
  - Iterations: 12500
  - Data augmentation: Random-resized-crop

- **Grid Search:**
  - Report highest accuracy on validation set

- **Inference Cost:**
  - Single inference on backbone per iteration
  - Single matrix multiplication per linear classifier

## C List of Datasets used

**Datasets and Their Uses:**

- **Image Classification:**
  - **ImageNet:**
    - ImageNet-1k: Pretrained, Retrieved, Evaluated for Classification (Russakovsky et al., 2015)
    - ImageNet-22k: Pretrained, Retrieved (Deng et al., 2009)
    - ImageNet-V2, ImageNet-ReaL, ImageNet-A, ImageNet-C, ImageNet-R, ImageNet-Sk: Evaluated for Classification (Recht et al., 2019; Beyer et al., 2020; Hendrycks et al., 2021a, 2021b; Hendrycks & Dietterich, 2019; Wang et al., 2019)
  - **Other Datasets:**
    - Food-101, CIFAR-10, CIFAR-100, SUN397, StanfordCars, FGVC-Aircraft, VOC 2007, DTD, Oxford Pets, Caltech101, Flowers, CUB200: Pretrained, Retrieved, Evaluated for Classification (Bossard et al., 2014; Krizhevsky et al., 2009; Xiao et al., 2010; Krause et al., 2013; Maji et al., 2013; Everingham et al., 2010; Cimpoi et al., 2014; Parkhi et al., 2012; Fei-Fei et al., 2004; Nilsback & Zisserman, 2008; Welinder et al., 2010)
    - iNaturalist 2018, iNaturalist 2021, Places-205: Evaluated for Classification (Van Horn et al., 2018, 2021; Zhou et al., 2014)

- **Video Classification:**
  - UCF101, Kinetics-400, SSv2: Evaluated for Video Classification (Soomro et al., 2012; Kay et al., 2017; Goyal et al., 2017)

- **Retrieval:**
  - R-Paris, R-Oxford, Met, Amstertime: Retrieved, Evaluated for Retrieval (Radenović et al., 2018a; Ypsilantis et al., 2021; Yildiz et al., 2022)

- **Semantic Segmentation:**
  - ADE20k, Cityscapes, VOC 2012: Evaluated for Semantic Segmentation (Zhou et al., 2017; Cordts et al., 2016; Everingham et al., 2010)

- **Depth Estimation:**
  - NYU-Depth V2, KITTI, SUN-RGBD: Evaluated for Depth Estimation (Silberman et al., 2012; Geiger et al., 2013; Song et al., 2015)

- **Fairness:**
  - DollarStreet, Casual Conv.: Evaluated for Fairness (De Vries et al., 2019; Hazirbas et al., 2021)

- **Pretraining:**
  - GLD v2: Pretrained (Weyand et al., 2020)
  - Mapillary SLS: Pretrained (Warburg et al., 2020)
