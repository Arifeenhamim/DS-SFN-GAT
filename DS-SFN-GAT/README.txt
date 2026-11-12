DS-SFN-GAT MANUSCRIPT: CODE DOCUMENTATION AND REPRODUCIBILITY STATUS
====================================================================

STATUS
======
This README documents the supplied file DS-SFN-GAT/DS-SFN-GAT.ipynb and identifies the remaining requirements for the DS-SFN-GAT submission. It is a working README, not a claim that the manuscript results are reproducible from this notebook.

The notebook's training implementation is CLIPFFTDetector, a frozen CLIP image encoder plus Fourier features and gated fusion. It is NOT the residual spatial encoder, spectral convolutional encoder, attention fusion, and reconstruction branch described as DS-SFN-GAT in the manuscript. The matching proposed-model implementation must be supplied before this package can serve as the final journal code supplement.

No training was rerun when preparing this documentation. Reported notebook numbers below are retained outputs, not independently reproduced results.

REPOSITORY AND VERSION
======================
Repository: https://github.com/Arifeenhamim/DS-SFN-GAT
Code folder: https://github.com/Arifeenhamim/DS-SFN-GAT/tree/main/DS-SFN-GAT
Notebook: DS-SFN-GAT/DS-SFN-GAT.ipynb
Documented commit: b9bdd30ff6f643c60583395f08c44240ad2fb562
Notebook SHA-256: 0b4519c8b50217004f0c7cd3ef74003f2ace9ef2ab91ad92fdf8d9377be5a15f

The inspected repository contains one notebook. No requirements file,
standalone model scripts, checkpoint files, README, or LICENSE file were
present at this commit. This README.txt is a separate deliverable and has
not been committed to the repository.

PROJECT DESCRIPTION
===================
Associated manuscript: "DS-SFN-GAT: A Dual-Stream Spatial-Frequency Network with Graph Attention Fusion for Robust Detection of AI-Generated Images."

Task: binary classification of real and AI-generated images.
Available implementation: CLIP–FFT with learned gated feature fusion.

DATASET INFORMATION
===================
CIFAKE: Real and AI-Generated Synthetic Images.
Original source: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
Authors' repository: https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images
Publication: https://doi.org/10.1109/ACCESS.2024.3356122

CIFAKE contains 120,000 colour images at 32 × 32 pixels: 60,000 real images originating from CIFAR-10 and 60,000 synthetic images generated with Stable Diffusion v1.4.

Required directories:
- train/REAL/
- train/FAKE/
- test/REAL/
- test/FAKE/

The notebook pools the original train and test folders, sorts filenames within each folder, and creates a new stratified split with seed 42:
- Training: 84,000 images (42,000 per class).
- Validation: 18,000 images (9,000 per class).
- Test: 18,000 images (9,000 per class).

Labels are REAL = 0 and FAKE = 1. This is a custom partition, not CIFAKE's original benchmark partition. Training augmentations are applied online and do not create additional independent source images.

The notebook does not include the GenImage experiments reported in the manuscript. If those results are retained, release their source URLs, subset identifiers, split manifests, and corresponding code.

CODE INFORMATION
================
Cell numbers below are one-based, counting all notebook cells:
1. Installs OpenAI CLIP.
2. Defines data loading, transforms, CLIPFFTDetector, training, validation, checkpointing, test evaluation, and plots.
3. Displays smoothed training and validation curves.
4. Contains super-resolution and explanation routines for a different model interface.
5. Contains a saliency routine dependent on cell 4.
6. Prints an explanation heading.
7. Empty cell.

Core classes:
- CIFAKEDataset: loads RGB images and binary labels.
- FFTFeatureExtractor: averages the normalized input over channels, applies a two-dimensional FFT, takes log(abs(FFT) + 1e-6), and projects flattened features to 256 dimensions.
- CLIPEncoder: loads the frozen OpenAI CLIP ViT-B/32 visual encoder.
- GatedFusion: projects the two feature vectors and combines them using a learned sigmoid gate.
- CLIPFFTDetector: produces one binary classification logit.
- EarlyStopping: monitors validation ROC-AUC with patience 7 and minimum improvement 1e-4.

REQUIREMENTS
============
Core imported packages:
- Python and a Jupyter-compatible notebook environment
- PyTorch and torchvision
- NumPy, SciPy, scikit-learn
- Pillow, Matplotlib, Seaborn
- tqdm and notebook widget support
- OpenAI CLIP from https://github.com/openai/CLIP

Git and internet access are needed for installation and the first CLIP weight download. A CUDA-capable GPU is recommended. Saved notebook output identifies a Tesla T4 with approximately 15.6 GB reported memory.

Exact package versions are not recorded in the supplied notebook. Its code uses numpy.dtypes, numpy._core.multiarray, and torch.serialization.add_safe_globals, so compatibility must be checked in the final tested environment. The saved installation output also contains dependency conflicts; it should not be treated as a clean environment specification.

The later, currently incompatible explanation cells additionally use OpenCV, timm, einops, SHAP, and external SwinIR code and weights.

USAGE INSTRUCTIONS FOR THE AVAILABLE CLIP–FFT EXPERIMENT
========================================================
These are source-derived instructions, not a verified fresh execution.

1. Obtain the documented revision:
   git clone https://github.com/Arifeenhamim/DS-SFN-GAT.git
   cd DS-SFN-GAT
   git checkout b9bdd30ff6f643c60583395f08c44240ad2fb562

   Open DS-SFN-GAT/DS-SFN-GAT.ipynb in Kaggle or Jupyter.
   In Kaggle, upload this notebook, attach CIFAKE, enable internet for
   dependency/weight downloads, and select a GPU runtime if available.
2. Attach/download CIFAKE and preserve the directory layout above.
3. Set DATASET_ROOT in the main training cell to the extracted dataset root.
   The supplied path is:
   /kaggle/input/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
4. Set writable OUTPUT_DIR and CKPT_DIR paths. Defaults are:
   /kaggle/working/outputs
   /kaggle/working/checkpoints
5. Install compatible PyTorch/torchvision for the selected runtime.
   For a local environment, create and activate a virtual environment
   using your platform's standard procedure. Then install the remaining
   core dependencies:

   python -m pip install numpy scipy scikit-learn pillow matplotlib seaborn tqdm jupyterlab ipywidgets
   python -m pip install git+https://github.com/openai/CLIP.git
   python -m jupyterlab

   Alternatively execute the first notebook cell to install CLIP.
   These commands are setup guidance, not a tested or version-pinned
   environment. Resolve compatibility in a clean environment and record
   the successful versions before submission:

   python -m pip freeze > requirements.txt

   Also record Python, CUDA, PyTorch, GPU/driver versions, and the CLIP
   commit. Installing CLIP from an unpinned branch may change dependencies.
6. Restart the kernel if installation changes imported packages.
7. Execute the main training cell. This downloads CLIP weights when necessary, creates the custom split, trains the model, selects a checkpoint on validation ROC-AUC, and evaluates it on the test subset.
8. Optionally execute the curve-smoothing cell. Smoothed curves are presentation aids; retain raw histories for interpretation.
9. Do NOT use Run All as a reproducibility procedure for the current notebook: the later explanation cells are incompatible with the trained model.

AVAILABLE IMPLEMENTATION METHODOLOGY
====================================
Training images are resized to 224 × 224, randomly flipped horizontally, colour-jittered (brightness 0.2, contrast 0.2, saturation 0.1), converted to tensors, and normalized using:
- Mean: (0.48145466, 0.4578275, 0.40821073)
- Standard deviation: (0.26862954, 0.26130258, 0.27577711)

Validation/test images use resizing, tensor conversion, and the same normalization without random augmentation.

Training configuration:
- Maximum epochs: 40
- Batch size: 256
- Data-loader workers: 2
- Optimizer: AdamW
- Initial learning rate: 0.0003
- Weight decay: 0.01
- Schedule: CosineAnnealingLR
- Loss: BCEWithLogitsLoss (no reconstruction loss)
- Dropout: 0.3
- Gradient norm clipping: 1.0
- Seed: 42 for Python, NumPy, and PyTorch
- CUDA automatic mixed precision when available; no explicit bfloat16 selection
- Checkpoint: highest observed validation ROC-AUC
- Early stopping: patience 7, minimum ROC-AUC improvement 0.0001
- Predictions: FAKE if sigmoid(logit) > 0.5; otherwise REAL

The checkpoint rule and early-stopping improvement rule are not identical: the checkpoint accepts any higher AUC, while early stopping uses a minimum improvement.

Metrics are accuracy, binary precision, recall, F1, ROC-AUC, average precision, and Matthews correlation coefficient. Average precision is computed with sklearn.metrics.average_precision_score; it is not automatically equivalent to trapezoidal PR-AUC.

OUTPUTS
=======
Checkpoint:
- checkpoints/clip_fft_best.pt (epoch, state dictionary, validation metrics)

Figures:
- outputs/training_curves.png
- outputs/roc_pr_curves.png
- outputs/confusion_matrix.png
- outputs/score_distribution.png

Test metrics are printed in the notebook. The supplied code does not export a complete per-image prediction table or split manifests.

Saved test output:
- Accuracy: 0.9744
- F1: 0.9743
- Precision: 0.9770
- Recall: 0.9717
- ROC-AUC: 0.9971
- Average precision: 0.9971
- MCC: 0.9488

These numbers belong to the available CLIP–FFT experiment and must not be attributed to DS-SFN-GAT.

ALGORITHM SUMMARY FOR THE AVAILABLE CODE
========================================
1. Load and label real/fake image paths from both original dataset folders.
2. Create reproducible stratified training, validation, and test partitions.
3. Apply training-only augmentation and deterministic evaluation transforms.
4. Extract a frozen CLIP visual embedding and a trainable FFT embedding.
5. Project both embeddings and combine them with a learned sigmoid gate.
6. Produce a classification logit and optimize binary cross-entropy.
7. Monitor validation ROC-AUC and save the highest-AUC checkpoint.
8. Stop according to the configured patience or maximum epoch count.
9. Reload the selected checkpoint and calculate metrics on the test set.
10. Save diagnostic plots and retain the printed test measurements.

KNOWN LIMITATIONS AND REQUIRED COMPLETION
=========================================
Before journal submission:
1. Supply the original DS-SFN-GAT implementation and its execution instructions.
2. Reconcile architecture, preprocessing, hardware, checkpoint selection, and outputs with the manuscript.
3. Supply the code and logs for every retained baseline, ablation, cross-validation, corruption, sensitivity, and generator experiment.
4. Export the tested dependency versions, split manifests, configuration, raw training histories, and per-image predictions.
5. Include checkpoints where practical and document the exact code version.
6. Test the final package from a fresh kernel/environment.
7. Replace this working status section with an accurate description of the completed package.

Explanation cells currently expect model.spatial_bb and dictionary inputs with spatial/spectral/raw tensors. The available CLIPFFTDetector instead takes a tensor and does not expose spatial_bb. Explanation preprocessing also differs from the manuscript. Saved explanation outputs do not establish that these cells run sequentially with the supplied training code.

CITATIONS
=========
- Bird, J. J., and Lotfi, A. (2024). CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. IEEE Access. https://doi.org/10.1109/ACCESS.2024.3356122
- Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images. https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
- Cite the associated manuscript using its final publication details when available.
- Credit OpenAI CLIP and any third-party software or weights included in the final package.

LICENSE AND CONTRIBUTION GUIDELINES
===================================
No author-selected license for the research code was supplied. Add an explicit LICENSE file before public release; this README does not grant a code license.
Third-party data, code, and pretrained weights retain their own licensing terms. Preserve required attribution and notices.
Report reproducibility issues through:
https://github.com/Arifeenhamim/DS-SFN-GAT/issues
or contact the manuscript's corresponding author. Include the code version,
environment, configuration, command/cell, and complete error message.
Proposed contributions should explain their purpose and validation and
must not silently replace experimental outputs. No repository-specific
contribution policy was present in the inspected revision.

