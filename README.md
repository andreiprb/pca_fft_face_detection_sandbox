# SVM Face Detection Sandbox

Face recognition models (starting point):
- sklearn model: https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html
- pycodemates model: https://www.pycodemates.com/2022/12/svm-for-face-recognition-using-python.html
- mlmastery model: https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/
- madeyoga model: https://github.com/madeyoga/face-recognition

Benchmarks:
- OpenCV Cascade Classifier (Viola-Jones)
- Dlib SVM & HOG model
- YOLOv5, YOLOv8

# TODOs:

Dataset:
- negative images are too small (32x32), LFW images are 250x250 natively
- explore FDDB dataset or find some similar dataset to CIFAR-10 for the negative class, with bigger images
- explore CelebA dataset (https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- Current datasets:
  - Labeled Faces in the Wild (LFW) - 13,000 images of faces
    - https://complexity.cecs.ucf.edu/lfw-labeled-faces-in-the-wild/
    - https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
  - CIFAR-10 - 60000 non-face images
    - https://www.cs.toronto.edu/~kriz/cifar.html
    - https://www.kaggle.com/c/cifar-10/

Feature extraction:
- some basic intuition suggests that the red channel is the most important for face detection
- explore PCA on green and blue channels, keeping the red channel as is, or some more complex combination of the three channels
- explore Hough transform

HOG:
- skimage HOG is painfully slow, ~4 minutes for load_dataset to execute, but provides decent results
- opencv HOG is significantly faster, ~3 seconds for load_dataset to execute, but doesn't work well (see visualization in sandbox.ipynb)
- explore using opencv HOG with different data preprocessing and parameter configurations to get closer to skimage current implementation (from sandbox.ipynb)
- optionally, implement our own HOG function to get more control
- explore combining HOG with edge detection, see whether PCA works better there

PCA and feature selection:
- use PCA for selecting eigenfaces for FFT

SVM:
- explore SW + SVM in case region proposal yields low confidence

Detection algorithm:
- sliding_window is painfully slow
- explore using FFT for region proposal
