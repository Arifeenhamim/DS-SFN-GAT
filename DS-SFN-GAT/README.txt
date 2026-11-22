DS-SFN-GAT project | CLIP–FFT image detection on CIFAKE
======================================================

This notebook trains a binary classifier to distinguish real images from
AI-generated images in CIFAKE. It contains the project's CLIP–FFT experiment.
The graph-attention model described in the associated DS-SFN-GAT manuscript
uses a separate architecture.

Dataset
-------
Download CIFAKE from Bird and Lotfi:
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

Set DATASET_ROOT to the directory containing train/ and test/. Both directories
must contain REAL/ and FAKE/ subdirectories. The notebook labels REAL as 0 and
FAKE as 1. It pools the original directories and creates a stratified split
with seed 42: 84,000 training images, 18,000 validation images and 18,000
test images. Each split contains equal numbers of real and synthetic images.

Running the notebook on Kaggle
------------------------------
1. Import DS-SFN-GAT/DS-SFN-GAT.ipynb into Kaggle. Attach the CIFAKE dataset,
   select a GPU accelerator and enable internet access for the CLIP installation
   and pretrained weights.
2. Run the first cell to install OpenAI CLIP. The training cell also imports
   PyTorch, torchvision, NumPy, SciPy, scikit-learn, Pillow, Matplotlib,
   Seaborn and tqdm. The progress bars use notebook widgets.
3. In the second cell, set DATASET_ROOT to the path shown for the attached
   dataset. The default output locations are /kaggle/working/outputs for
   figures and /kaggle/working/checkpoints for model weights.
4. Run the second cell to train and evaluate the classifier. It prints the
   final test metrics after reloading the checkpoint with the highest
   validation ROC-AUC. Run the third cell in the same session if you also
   want to display smoothed training curves. The CLIP–FFT workflow ends here.
   The later explanation cells expect a different model interface.

Model and training
------------------
Images are resized to 224 x 224 pixels. Training also uses random horizontal
flips and colour jitter. A frozen CLIP ViT-B/32 encoder extracts visual
features. A second branch computes a two-dimensional Fourier transform and
projects the log-magnitude spectrum to a feature vector. A learned gate
combines the two vectors for binary classification. The notebook trains with
AdamW and binary cross-entropy for up to 40 epochs, with early stopping
based on validation ROC-AUC.

Outputs
-------
checkpoints/clip_fft_best.pt   Selected checkpoint
outputs/training_curves.png    Training and validation curves
outputs/roc_pr_curves.png      ROC and precision–recall curves
outputs/confusion_matrix.png   Test-set confusion matrix
outputs/score_distribution.png Test prediction score distributions

References and support
----------------------
Bird, J. J. and Lotfi, A. (2024). CIFAKE: Image Classification and
Explainable Identification of AI-Generated Synthetic Images. IEEE Access.
https://doi.org/10.1109/ACCESS.2024.3356122

OpenAI CLIP code and pretrained weights:
https://github.com/openai/CLIP

For questions about running the notebook, open a repository issue and include
the cell number and error message:
https://github.com/Arifeenhamim/DS-SFN-GAT/issues
