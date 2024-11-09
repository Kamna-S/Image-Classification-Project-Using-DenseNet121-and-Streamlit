# Industrial Equipment Defect Classification Using DenseNet121

## Project Overview
This project involves classifying images of industrial equipment into two categories: 'Defective' and 'Non-defective'. The model uses the DenseNet121 deep learning architecture, a pre-trained Convolutional Neural Network (CNN), and is fine-tuned for binary classification. The application aims to assist in quality control by automating the defect detection process.

## Dataset
The dataset consists of images of industrial equipment, labeled as:
- **Defective**
- **Non-defective**

For defective images, additional labels specifying the type of defect could be included for further analysis. Data augmentation techniques such as rotation, flipping, and zooming were applied to improve the model's generalization.

## Model Architecture
- **Model**: DenseNet121
- **Framework**: TensorFlow 2.x (Keras)
- **Pre-trained on**: ImageNet dataset
- **Final layer**: Modified for binary classification

The model was trained using binary cross-entropy loss and the Adam optimizer. The training was conducted with an 80-20 training-validation split.

## Dataset Link
https://drive.google.com/drive/folders/1EEdsVtKJIeLKEjnkGMqcO5P6BI6cxW5K?usp=sharing

##Images

<img width="294" alt="Screenshot 2024-11-09 204304" src="https://github.com/user-attachments/assets/aa0b09c6-323a-490a-a55e-6f9c5b55193a">

<img width="389" alt="Screenshot 2024-11-09 204103" src="https://github.com/user-attachments/assets/3cf0c2f4-ad4a-45eb-8cef-5c38fd4660c9">

##Demo Video Link
https://drive.google.com/file/d/1uKSlVOQ51gEpf-etFEmITaQKyxpqzlgb/view?usp=drive_link


