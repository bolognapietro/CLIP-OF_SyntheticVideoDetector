# Optical Flow and Frame Analysis Framework

## Overview
This repository provides a framework for detecting synthetic videos by combining information from multiple feature extraction techniques. The system integrates semantic high-level features (via CLIP), low-level pixel artifacts (via Corvi2023), and motion analysis (via RAFT-based optical flow) to build a robust detector that outperforms existing methods.

## Key Features
- **High-Level Semantic Features**: extracted using pre-trained *CLIP* models for semantic content analysis.
- **Low-Level Pixel Artifacts**: detected with *Corvi2023* for identifying inconsistencies in pixel distributions.
- **Motion Analysis**: *RAFT-based optical flow* to capture motion artifacts and temporal inconsistencies.
- **Fusion-Based Detection**: combines predictions from the three feature types using advanced fusion techniques to improve detection accuracy.
- **Performance Comparison**: evaluates and compares the new detector against existing methods to validate its superiority.

## Repository Structure
- `main.py`: main script for orchestrating video analysis, including frame extraction, feature fusion, and evaluation.
- `optical_flow/optical_flow.py`: module for RAFT-based optical flow generation and motion-related feature extraction.
- `weights/`: Pre-trained weights and configurations for CLIP, Corvi2023, and RAFT models.
- `results/`: Directory for storing processed results, including intermediate outputs and final predictions.
- `dataset/`: Folder containing the video dataset to be analyzed.

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### Process Videos
To start processing videos and evaluate the synthetic video detector:

```bash
python main.py
```
You will be prompted to select one of the following options:

 1. Process Luma videos.
 
 2. Process Real videos.
 
 3. Process CogVideoX-5b videos.
 
 4. Process Hunyuan videos.
 
 5. Process all videos in the dataset.

---
### Feature Extraction
- Frames: Frames are automatically extracted from the videos during processing.
- Semantic Features: Extracted using CLIP models from the weights/ directory.
- Pixel Artifacts: Detected using Corvi2023 weights and configurations.
- Motion Features: Generated through RAFT optical flow analysis.

---
### Fusion and Prediction
The system uses multiple fusion methods (mean, max, soft probabilities, etc.) to aggregate predictions from semantic, pixel, and motion features. Results are stored as CSV files in the `results/` directory.

> **⚠️ N.B.**
> Based on our experiments, the optimal fusion method for combining Corvi2023 and CLIP predictions is the **soft OR probability**. To incorporate the Optical Flow (OF) contribution, we recommend using a **simple mean probability** to fuse the outputs of these two stages with motion-based features, achieving the best overall performance.

---
### Evaluate Performance
The system outputs performance metrics comparing the new detector against baseline methods. This helps validate the improved accuracy of the combined approach.

---
### Expected Output
For each video, the system generates:

- Frame-by-frame predictions given semanthic and pixel features (`<type_of_run>/frames_results_<filename>.csv`, where `type_of_run` may be [`complete_dataset`, `luma`, `hunyuan`, `real`, `cogvideo`] depending on what you choose to process on the terminal)
- Video prediction given semanthic, pixel and motion (optical flow) features (`results_<type_of_run>.csv`, where `type_of_run` may be [`complete_dataset`, `luma`, `hunyuan`, `real`, `cogvideo`] depending on what you choose to process on the terminal)
  - In this file, if prediction > 0.5 the video is considered to be synthetic (OF contributions have another scale, we will fuse them after).

---
### Graph Creation and Performance Analysis
To visually represent the results obtained in a clearer and more comprehensible way, we have developed a set of scripts for graph creation. Follow these steps to generate the graphs:

1. Navigate to the _graph_creation_ directory by running:
   ```bash
   cd graph_creation 
   ```
2. The directory contains four different scripts, each designed to create a specific type of graph. All scripts use the CSV file `results_complete_dataset.csv` as input and generate corresponding graphs:
   - `detector_graph_maker.py`: generates a graph to analyse the performance of the two baseline detectors, _CLIP_ and _AIGVDet_. 
   - `detector_graph_maker_category.py`: similar to the above, but allows for performance comparisons on a category-by-category basis.
   - `CLIP-OF_graph_maker.py`: creates a graph to evaluate the performance of our model, _CLIP-OF_. 
   - `detector_graph_maker_category.py`: similar to the above, but enables performance evaluation by individual categories.

The folder `graph_creation/results` contains the most significant results, organized for easy reference. Specifically, it includes:
- __CLIP Performance Graph__: `prediction_fusion[soft_or_prob] results.png`
- __AIGVDet Performance Graph__: `prediction_OF results.png`
- __CLIP-OF Performance Graph__: `CLIP-OF results.png`

Additionally, the folder provides all category-by-category performance graphs for both CLIP and CLIP-OF models.

Here, we also provide a guide to read the graphs correctly:
 - The __x-axis__ represents the samples grouped into categories (e.g., CogVideoX, Luma, HunyuanVideo, Real)
 - The __y-axis__ shows the prediction scores ranging from 0.0 to 1.0. 
 - Each sample is represented by a __blue dot__, with blue vertical lines connecting the dots to the baseline (0.0) for clarity.
 - The __red dashed__ line indicates the decision threshold:
   - The __red area__ represents the prediction that are classified as _Fake Videos_.
   - The __green area__ represent the prediction that are classified as _Real Videos_.
- Below each category, the graph includes an __accuracy summary__ showing correct predictions over the total number of samples.

These graphs provide a clear view of the prediction distribution, highlighting false positives, false negatives, and performance across categories.
<p align="center" text-align="center"> 
    <img width="90%" src="graph_creation/results/CLIP-OF results.png"> 
    <br> 
    <span><i>Example of a performance analysis graph</i></span> 
</p>

---
### Research Objective
This project aims to build a synthetic video detector that leverages high-level semantics, low-level pixel details, and motion analysis to achieve superior detection performance compared to existing methods.

---
## Acknowledgments
This framework builds upon the foundations of several state-of-the-art models:

- **RAFT**: For optical flow-based motion analysis.
- **CLIP**: For high-level semantic feature extraction.
- **LCIP and Corvi2023**: For detecting low-level pixel artifacts, developed by Cozzolino et al. at the University of Naples.
- **AIGVDet**: A framework for AI-generated video detection, integrating semantic, artifact-based, and motion features.

---
### Citations
If you use this framework in your research, please consider citing the following works:

1. **"RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"**  
   *Authors*: Zachary Teed, Jia Deng  
   *Conference*: European Conference on Computer Vision (ECCV), 2020  
   *Abstract*: RAFT introduces a novel approach for dense optical flow estimation based on a recurrent neural network architecture. It sets new benchmarks for accuracy and speed in optical flow tasks.  
   [Link to Paper](https://arxiv.org/abs/2003.12039) | [GitHub Repository](https://github.com/princeton-vl/RAFT)

2. **"Raising the Bar of AI-generated Image Detection with CLIP"**  
   *Authors*: Davide Cozzolino, Giovanni Poggi, Riccardo Corvi, Matthias Nießner, and Luisa Verdoliva  
   *Conference*: IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2024  
   *Abstract*: This work explores the potential of pre-trained vision-language models (VLMs) for universal detection of AI-generated images. The method demonstrates significant improvements in generalization and robustness against various image impairments.  
   [Link to Paper](https://arxiv.org/abs/2312.00195) | [GitHub Repository](https://github.com/grip-unina/ClipBased-SyntheticImageDetection)

3. **"On the Detection of Synthetic Images Generated by Diffusion Models"**  
   *Authors*: Riccardo Corvi, Davide Cozzolino, Giada Zingarini, Giovanni Poggi, Koki Nagano, and Luisa Verdoliva  
   *Conference*: IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2023  
   *Abstract*: This paper investigates the challenges of distinguishing synthetic images generated by diffusion models from pristine ones, focusing on forensic traces and robustness in social-network scenarios.  
   [Link to Paper](https://arxiv.org/abs/2312.00195) | [GitHub Repository](https://github.com/grip-unina/DMimageDetection)

4. **"AIGVDet: AI-Generated Video Detection with Cross-Modal Feature Fusion"**  
   *Authors*: Lin Zhang, Xiaoyu Qiao, Yadong Wu, Yiming Li, Can He, Honggang Zhang, Xiaonan Guo, Xiaochun Cao  
   *Conference*: [Include Conference, if applicable]  
   *Abstract*: AIGVDet integrates high-level semantic features, low-level artifacts, and motion features for detecting AI-generated videos. It achieves state-of-the-art performance by leveraging a cross-modal feature fusion strategy.  
   [Link to Paper](https://arxiv.org/pdf/2403.16638v1) | [GitHub Repository](https://github.com/multimediaFor/AIGVDet)

We extend our gratitude to the open-source contributors and research communities that made these tools available.




