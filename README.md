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
- explore FDDB dataset 
- explore CelebA dataset (https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) - ~200k faces
- Current datasets:
  - Labeled Faces in the Wild (LFW) - ~13k faces
    - https://complexity.cecs.ucf.edu/lfw-labeled-faces-in-the-wild/
    - https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

Feature extraction:
- PCA (SVD) for selecting eigenfaces for region proposal (DCT/FFT)
- some basic intuition suggests that the red channel is the most important for face detection
- explore PCA on green and blue channels, keeping the red channel as is, or some more complex combination of the three channels
- explore Hough transform

HOG:
- skimage HOG is painfully slow, ~4 minutes for load_dataset to execute, but provides decent results
- opencv HOG is significantly faster, ~3 seconds for load_dataset to execute, but doesn't work well (some weird artefacts are generated, to investigate)
- explore using opencv HOG with different data preprocessing and parameter configurations to get closer to skimage current implementation
- explore using personal HOG implementation

Region proposal:
- sliding window is painfully slow
- explore using DCT/FFT for region proposal

SVM:
- explore focused sliding window + SVM in case region proposal yields low confidence
