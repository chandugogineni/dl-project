# Semantic Segmentation of Medical Images

## Team Members
- Lakshmi Prasanna Doupati  
- Chandu Gogineni  

## Project Overview
This project explores semantic segmentation techniques for medical images, specifically targeting **breast cancer ultrasound images** and **dental panoramic X-rays**. The study compares two deep learning-based models: **U-Net** and a **VGG16-based model**, analyzing their effectiveness in segmenting regions of interest within these medical images.

## Contents
- Abstract
- Introduction
- Methodology
- Training Results
- Conclusion
- References

## Abstract
Medical image segmentation is crucial for assisting clinicians in diagnostics and treatment planning. This project implements and compares **U-Net** and **VGG16-based models** for segmenting **breast cancer ultrasound images** and **dental panoramic X-rays**. The models are evaluated based on segmentation accuracy, computational efficiency, and potential real-world applications. The results show that both models achieve competitive segmentation accuracy, providing valuable insights for automated medical imaging analysis.

## Methodology
### Data Acquisition & Preprocessing
- **Datasets Used:**
  - **Breast Cancer Ultrasound Images Dataset** ([gymprathap/Breast-Cancer-Ultrasound-Images-Dataset](https://huggingface.co/datasets/gymprathap/Breast-Cancer-Ultrasound-Images-Dataset))
  - **Teeth Segmentation in Panoramic X-ray Dataset** ([SerdarHelli/SegmentationOfTeethPanoramicXRayImages](https://huggingface.co/datasets/SerdarHelli/SegmentationOfTeethPanoramicXRayImages))
- **Preprocessing:**
  - Image size normalization
  - Noise reduction
  - Data augmentation (random flipping, rotation, cropping)

### Model Architectures
#### U-Net Model
- Encoder-decoder architecture with **skip connections** for better feature localization.
- Suitable for biomedical image segmentation.
- Uses **convolutional and upsampling layers** to reconstruct segmented regions.

#### VGG16-Based Model
- Leverages **VGG16 as a feature extractor**.
- Pretrained on ImageNet, with modified fully connected layers replaced by convolutional layers.
- Skip connections help improve segmentation performance.

### Training & Optimization
- **Loss Function:** Dice coefficient loss & Binary Cross-Entropy (BCE)
- **Optimizer:** Adam optimizer
- **Evaluation Metrics:**
  - Accuracy
  - Intersection over Union (IoU)
  - Precision & Recall
  
## Training Results
- **Breast Cancer Segmentation:**
  - **U-Net** achieved slightly higher accuracy than **VGG16-based model**.
  - VGG16-based model leveraged pre-trained weights for competitive results.
- **Dental X-ray Segmentation:**
  - U-Net effectively segmented dental structures.
  - VGG16-based model showed comparable performance with lower computational requirements.

## Conclusion
- Both models effectively segment medical images, aiding in **clinical decision-making**.
- U-Net demonstrated superior accuracy, while **VGG16-based model** provided efficiency advantages.
- Future work includes enhancing **dataset diversity, exploring ensemble methods, and deploying models in real-world medical settings**.

## References
1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
2. Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition.
3. Tajbakhsh et al. (2016). Breast Cancer Segmentation Using Convolutional Neural Networks.
4. Guo et al. (2020). Automatic Breast Cancer Segmentation Using Deep Learning Techniques with Ensemble Methods.
5. Feng et al. (2022). Teeth Segmentation in Panoramic Dental X-ray Using Mask R-CNN.
6. Moin et al. (2020). Segmentation of Dental Restorations on Panoramic Radiographs Using Deep Learning.

## Usage
To run the models, follow these steps:
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/semantic-segmentation-medical-images.git
   cd semantic-segmentation-medical-images
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Train the model:
   ```sh
   python train.py --model unet
   ```
4. Evaluate the model:
   ```sh
   python evaluate.py --model vgg16
   ```
5. Visualize results:
   ```sh
   python visualize.py
   ```

## Future Work
- **Enhancing dataset diversity** for better generalization.
- **Exploring ensemble techniques** for improved segmentation accuracy.
- **Deploying models in healthcare applications** for real-time usage.

## Contact
For any inquiries, please contact:
- **Lakshmi Prasanna Doupati** - Email: example@email.com
- **Chandu Gogineni** - Email: example@email.com

