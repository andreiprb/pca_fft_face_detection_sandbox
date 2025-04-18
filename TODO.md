# TODOs:

Dataset:
- negative images too small (32x32, resized to 64x64), LFW images 250x250 natively, losing lots of data from a window size of 64x64
- explore FDDB dataset or find some similar dataset to CIFAR-10 for negative class, with bigger images

Feature extraction:
- some basic intuition suggests that the red channel is the most important for face detection
- explore pca on green and blue channels, keeping the red channel as is, or some more complex combination of the three channels
- explore Hough transform

HOG:
- skimage HOG is painfully slow, ~4 minutes for load_dataset to execute, but provides decent results
- opencv HOG is significantly faster, ~3 seconds for load_dataset to execute, but doesn't work well
- explore using opencv HOG with different data preprocessing and parameter configurations to get closer to skimage current implementation (from sandbox.ipynb)
if needed, implement our own HOG function to get more control
- explore combining HOG with edge detection, see whether PCA works better there

PCA and feature selection:
- for now, PCA generally makes predictions worse, but it is worth exploring further
- explore using PCA to reduce the number of features, but be careful not to lose too much information

SVM:
- explore other SVCs (poly, radial basis function) and hyperparameters
- explore multiclass SVCs approach for face feature detection (nose, chin, eyes, etc.) side by side with the current SVC

Detection algorithm:
- sliding_window is painfully slow
- explore using a Selective Search or EdgeBoxes approach to generate region proposals
- explore using a Quad Tree approach based on said region proposals
- explore using Disjoint Set Union for merging face features into a single face (SVM issue no. 2)
