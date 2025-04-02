import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_lfw_people, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from skimage.feature import hog
from typing import Tuple, List, Optional
import joblib, os
from skimage import data
from skimage.transform import resize


def extract_hog_features(images: np.ndarray, image_shape: Tuple[int, int],
                         pixels_per_cell: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Extract HOG (Histogram of Oriented Gradients) features from a set of images.

    HOG features are effective for object detection, particularly for faces, as they
    capture gradient structure that is characteristic of human faces.

    Args:
        images (np.ndarray): Array of images to extract features from
        image_shape (Tuple[int, int]): Height and width of each image
        pixels_per_cell (Tuple[int, int], optional): Size of cell for HOG computation.
            Defaults to (8, 8).

    Returns:
        np.ndarray: Array of HOG features for each image
    """
    features: List[np.ndarray] = []
    h: int
    w: int
    h, w = image_shape

    for image in images:
        if len(image.shape) == 1:
            image = image.reshape(h, w)

        hog_features: np.ndarray = hog(image, pixels_per_cell=pixels_per_cell,
                                       visualize=False, block_norm='L2-Hys')
        features.append(hog_features)

    return np.array(features)


def create_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Create a training dataset of face and non-face examples.

    This function:
    1. Loads the LFW dataset to get face examples
    2. Creates non-face examples from natural images datasets
    3. Splits the combined dataset into training and test sets

    Returns:
        Tuple containing:
            X_train (np.ndarray): Training data features
            X_test (np.ndarray): Test data features
            y_train (np.ndarray): Training data labels (1 for face, 0 for non-face)
            y_test (np.ndarray): Test data labels (1 for face, 0 for non-face)
            face_shape (Tuple[int, int]): Height and width of the face images
    """
    print("Loading LFW dataset...")
    lfw_people: any = fetch_lfw_people(min_faces_per_person=70, resize=0.5, color=False)
    face_images: np.ndarray = lfw_people.images

    n_samples: int
    h: int
    w: int
    n_samples, h, w = face_images.shape
    print(f"Dataset loaded: {n_samples} images of size {h}x{w}")

    face_labels: np.ndarray = np.ones(len(face_images))

    print("Creating non-face examples from natural images...")
    non_face_images: List[np.ndarray] = []

    # Method 1: Use scikit-image sample images
    sample_images = [
        data.astronaut(),
        data.camera(),
        data.coffee(),
        data.rocket(),
        data.chelsea(),
        data.clock(),
        data.coins(),
        data.horse(),
        data.hubble_deep_field(),
        data.immunohistochemistry(),
        data.logo(),
        data.page(),
        data.text(),
        data.checkerboard(),
    ]

    for img in sample_images:
        # Convert to grayscale if necessary
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Generate multiple non-face patches from each image
        for _ in range(n_samples // len(sample_images) + 1):
            if img.shape[0] > h and img.shape[1] > w:
                i = np.random.randint(0, img.shape[0] - h)
                j = np.random.randint(0, img.shape[1] - w)
                patch = img[i:i + h, j:j + w]
                non_face_images.append(patch)
            else:
                # Resize if the image is too small
                resized = resize(img, (h, w), anti_aliasing=True)
                if isinstance(resized, np.ndarray) and resized.ndim == 2:
                    non_face_images.append(resized)

    # Method 2: Load digits dataset (as additional non-face examples)
    print("Adding more non-face examples from digits dataset...")
    digits = load_digits()
    for digit_img in digits.images:
        # Resize digit to match face dimensions
        resized_digit = resize(digit_img, (h, w), anti_aliasing=True)
        non_face_images.append(resized_digit)

    # Method 3: Create texture patterns
    print("Generating texture patterns as additional non-face examples...")
    # Checkerboard patterns
    for scale in range(3, 20, 4):
        checkerboard = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                if (i // scale + j // scale) % 2 == 0:
                    checkerboard[i, j] = 1
        non_face_images.append(checkerboard)

    # Stripe patterns
    for scale in range(3, 20, 4):
        stripes = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                if (j // scale) % 2 == 0:
                    stripes[i, j] = 1
        non_face_images.append(stripes)

    # Add some pure random noise (but fewer than before)
    for _ in range(n_samples // 10):
        noise = np.random.rand(h, w)
        non_face_images.append(noise)

    # Method 4: Take crops from face images that don't contain key facial features
    print("Adding crops from LFW images that don't contain facial features...")
    for _ in range(n_samples // 2):
        random_idx = np.random.randint(0, len(face_images))
        img = face_images[random_idx]

        # Take crops from edges or corners where faces are less likely to be
        corners = [
            (0, 0),  # top-left
            (0, w - h // 3),  # top-right
            (h - h // 3, 0),  # bottom-left
            (h - h // 3, w - w // 3)  # bottom-right
        ]

        corner_idx = np.random.randint(0, len(corners))
        y, x = corners[corner_idx]
        crop_h = h // 3
        crop_w = w // 3

        if y + crop_h <= h and x + crop_w <= w:
            crop = img[y:y + crop_h, x:x + crop_w]
            crop_resized = cv2.resize(crop, (w, h))
            non_face_images.append(crop_resized)

    # Convert list to array and ensure all values are in [0, 1] range
    non_face_images_array = np.array(non_face_images)

    # Normalize images if needed
    for i in range(len(non_face_images_array)):
        if non_face_images_array[i].max() > 1.0:
            non_face_images_array[i] = non_face_images_array[i] / 255.0

    non_face_labels = np.zeros(len(non_face_images_array))

    print(f"Created {len(non_face_images_array)} non-face examples")

    # Balance the dataset if needed
    min_samples = min(len(face_images), len(non_face_images_array))
    face_indices = np.random.choice(len(face_images), min_samples, replace=False)
    non_face_indices = np.random.choice(len(non_face_images_array), min_samples, replace=False)

    face_subset = face_images[face_indices]
    non_face_subset = non_face_images_array[non_face_indices]
    face_labels_subset = face_labels[face_indices]
    non_face_labels_subset = non_face_labels[non_face_indices]

    # Flatten images for training
    X = np.vstack([
        face_subset.reshape(len(face_subset), -1),
        non_face_subset.reshape(len(non_face_subset), -1)
    ])
    y = np.hstack([face_labels_subset, non_face_labels_subset])

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of face examples: {np.sum(y == 1)}")
    print(f"Number of non-face examples: {np.sum(y == 0)}")

    # Visualize some examples
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        face_idx = np.random.choice(np.where(y == 1)[0])
        plt.imshow(X[face_idx].reshape(h, w), cmap='gray')
        plt.axis('off')
        plt.title('Face')

        plt.subplot(2, 10, i + 11)
        non_face_idx = np.random.choice(np.where(y == 0)[0])
        plt.imshow(X[non_face_idx].reshape(h, w), cmap='gray')
        plt.axis('off')
        plt.title('Non-face')

    plt.tight_layout()
    plt.savefig("results/dataset_examples.png")

    return X_train, X_test, y_train, y_test, (h, w)


# Modify the sliding window detection to be more selective
def sliding_window_detection(image: np.ndarray, model: Pipeline,
                             window_size: Tuple[int, int],
                             image_shape: Tuple[int, int],
                             step_size: int = 8,
                             scale_factor: float = 1.5,
                             min_scale: float = 0.8,
                             max_scale: float = 3.0,
                             confidence_threshold: float = 0.9) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect faces using a sliding window approach with a trained classifier.

    Modified to use a higher confidence threshold and more selective detection parameters.

    Args:
        image (np.ndarray): Input image
        model (Pipeline): Trained classifier pipeline
        window_size (Tuple[int, int]): Height and width of the detection window
        image_shape (Tuple[int, int]): Height and width used for HOG feature extraction
        step_size (int, optional): Step size for sliding window. Defaults to 8.
        scale_factor (float, optional): Factor between consecutive scales. Defaults to 1.5.
        min_scale (float, optional): Minimum scale to check. Defaults to 0.8.
        max_scale (float, optional): Maximum scale to check. Defaults to 3.0.
        confidence_threshold (float, optional): Minimum confidence for detection.
            Defaults to 0.9 (increased from original value of 0.8).

    Returns:
        List[Tuple[int, int, int, int, float]]: List of detections, each as
            (x, y, width, height, confidence)
    """
    gray: np.ndarray
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    detections: List[Tuple[int, int, int, int, float]] = []

    # Use fewer scales for more efficiency and precision
    scales: np.ndarray = np.geomspace(min_scale, max_scale, 5)  # geometric progression

    for scale in scales:
        resized: np.ndarray = cv2.resize(gray, (0, 0), fx=1 / scale, fy=1 / scale)

        if resized.shape[0] < window_size[0] or resized.shape[1] < window_size[1]:
            continue

        # Use larger step size for better performance
        for y in range(0, resized.shape[0] - window_size[0], step_size):
            for x in range(0, resized.shape[1] - window_size[1], step_size):
                window: np.ndarray = resized[y:y + window_size[0], x:x + window_size[1]]

                if window.shape[0] != window_size[0] or window.shape[1] != window_size[1]:
                    continue

                # Normalize window to match training data
                if window.max() > 1.0:
                    window = window / 255.0

                window_hog: np.ndarray = extract_hog_features([window], image_shape=window_size)

                confidence: float
                if hasattr(model, 'predict_proba'):
                    confidence = model.predict_proba(window_hog)[0][1]
                else:
                    pred: int = model.predict(window_hog)[0]
                    confidence = 1.0 if pred == 1 else 0.0

                if confidence > confidence_threshold:
                    actual_x: int = int(x * scale)
                    actual_y: int = int(y * scale)
                    actual_w: int = int(window_size[1] * scale)
                    actual_h: int = int(window_size[0] * scale)

                    detections.append((actual_x, actual_y, actual_w, actual_h, confidence))

    return detections


# Enhanced non_max_suppression with better IoU calculations
def non_max_suppression(detections: List[Tuple[int, int, int, int, float]],
                        overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int, float]]:
    """
    Apply Non-Maximum Suppression using IoU to remove redundant detections.

    This version uses a lower overlap threshold for more aggressive filtering.

    Args:
        detections (List[Tuple[int, int, int, int, float]]): List of detections, each as
            (x, y, width, height, confidence)
        overlap_threshold (float, optional): Threshold for overlap above which a detection
            is removed. Defaults to 0.3 (reduced from original value of 0.5).

    Returns:
        List[Tuple[int, int, int, int, float]]: Filtered list of detections.
    """
    if len(detections) == 0:
        return []

    # Extract box coordinates and confidence scores
    boxes: np.ndarray = np.array([d[:4] for d in detections])
    scores: np.ndarray = np.array([d[4] for d in detections])

    x1: np.ndarray = boxes[:, 0]
    y1: np.ndarray = boxes[:, 1]
    x2: np.ndarray = boxes[:, 0] + boxes[:, 2]
    y2: np.ndarray = boxes[:, 1] + boxes[:, 3]

    # Calculate area of each box
    area: np.ndarray = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort indices by confidence score, descending
    idxs: np.ndarray = np.argsort(scores)[::-1]

    pick: List[int] = []
    while len(idxs) > 0:
        # Select index with highest score
        i: int = idxs[0]
        pick.append(i)

        # Remove current index
        idxs = idxs[1:]
        if len(idxs) == 0:
            break

        # Calculate intersection coordinates between current box and remaining boxes
        xx1: np.ndarray = np.maximum(x1[i], x1[idxs])
        yy1: np.ndarray = np.maximum(y1[i], y1[idxs])
        xx2: np.ndarray = np.minimum(x2[i], x2[idxs])
        yy2: np.ndarray = np.minimum(y2[i], y2[idxs])

        # Calculate width and height of intersection
        w: np.ndarray = np.maximum(0, xx2 - xx1 + 1)
        h: np.ndarray = np.maximum(0, yy2 - yy1 + 1)
        intersection: np.ndarray = w * h

        # Calculate IoU for each remaining box
        iou: np.ndarray = intersection / (area[i] + area[idxs] - intersection)

        # Remove detections with IoU above threshold
        idxs = np.delete(idxs, np.where(iou > overlap_threshold)[0])

    return [detections[i] for i in pick]


# Functions to tune the model parameters
def optimize_detector_parameters(model: Pipeline, X_validation: np.ndarray, y_validation: np.ndarray) -> Pipeline:
    """
    Optimize parameters for the SVM detector to minimize false positives.

    Args:
        model (Pipeline): Trained model
        X_validation (np.ndarray): Validation data
        y_validation (np.ndarray): Validation labels

    Returns:
        Pipeline: Optimized model
    """
    # Extract the SVM from the pipeline
    if 'svc' in model.named_steps:
        svm = model.named_steps['svc']
        # Adjust decision threshold to favor precision over recall
        # Higher values of class_weight for class 0 will push the boundary
        # toward class 1, reducing false positives
        svm.class_weight = {0: 1.5, 1: 1.0}

    return model


def detect_faces(image_path: str, model: Pipeline,
                 face_shape: Tuple[int, int],
                 detector_type: str = 'svc') -> Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int, float]]]:
    """
    Detect faces in an image using SVC or KNN detector.

    This function loads an image, applies the face detector using a sliding window approach,
    filters the detections using non-maximum suppression, and visualizes the results.

    Args:
        image_path (str): Path to the input image
        model (Pipeline): Trained classifier pipeline
        face_shape (Tuple[int, int]): Height and width of the face images used for training
        detector_type (str, optional): Type of detector ('svc' or 'knn'). Defaults to 'svc'.

    Returns:
        Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int, float]]]:
            - Result image with detections visualized (or None if image cannot be loaded)
            - List of detections, each as (x, y, width, height, confidence)
    """
    image: Optional[np.ndarray] = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image at {image_path}")
        return None, []

    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detections: List[Tuple[int, int, int, int, float]] = sliding_window_detection(
        gray,
        model,
        window_size=face_shape,
        image_shape=face_shape,
        step_size=8,
        scale_factor=1.2,
        min_scale=0.8,
        max_scale=2.0,
        confidence_threshold=0.9  # Increased from 0.7 to reduce false positives
    )

    filtered_detections: List[Tuple[int, int, int, int, float]] = non_max_suppression(detections, overlap_threshold=0.3)

    print(f"Found {len(filtered_detections)} faces using {detector_type.upper()} detector")

    result_img: np.ndarray = image.copy()
    for (x, y, w, h, confidence) in filtered_detections:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result_img, f"{confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return result_img, filtered_detections


def save_detector(model: Pipeline, face_shape: Tuple[int, int], detector_type: str) -> None:
    """
    Save the trained detector to disk.

    This function saves the trained model and its metadata (face shape and detector type)
    to files that can be loaded later.

    Args:
        model (Pipeline): Trained classifier pipeline
        face_shape (Tuple[int, int]): Height and width of the face images used for training
        detector_type (str): Type of detector ('svc' or 'knn')
    """
    filename: str = f"models/{detector_type}_face_detector.joblib"
    joblib.dump(model, filename)

    with open(f"models/{detector_type}_detector_metadata.txt", "w") as f:
        f.write(f"Face shape: {face_shape[0]}x{face_shape[1]}\n")
        f.write(f"Detector type: {detector_type}\n")

    print(f"Detector saved as {filename}")


def load_detector(detector_type: str) -> Tuple[Pipeline, Tuple[int, int]]:
    """
    Load a saved detector from disk.

    This function loads a trained model and its metadata from files.

    Args:
        detector_type (str): Type of detector ('svc' or 'knn')

    Returns:
        Tuple[Pipeline, Tuple[int, int]]:
            - Loaded model
            - Face shape (height, width)
    """
    filename: str = f"models/{detector_type}_face_detector.joblib"
    model: Pipeline = joblib.load(filename)

    h: int
    w: int
    with open(f"models/{detector_type}_detector_metadata.txt", "r") as f:
        lines: List[str] = f.readlines()
        face_shape_str: str = lines[0].split(":")[1].strip()
        h, w = [int(x) for x in face_shape_str.split("x")]

    return model, (h, w)


def main(image: str) -> None:
    """
    Main function to train and test face detectors.

    This function:
    1. Creates a training dataset of face and non-face examples from diverse sources
    2. Trains SVC and KNN face detectors with optimized parameters
    3. Evaluates the detectors on test data
    4. Saves the trained detectors
    5. Tests them on a new image if available
    """
    # Create directories
    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists("results"):
        os.makedirs("results")

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    face_shape: Tuple[int, int]
    X_train, X_test, y_train, y_test, face_shape = create_training_data()

    # Split test set to have a validation set for parameter tuning
    X_test, X_validation, y_test, y_validation = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42)

    print("Training SVC face detector with optimized parameters...")
    svc_model: Pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=100, whiten=True, random_state=42),
        SVC(kernel='linear', probability=True, class_weight='balanced', C=10.0)
    )

    # Extract HOG features for training
    X_train_hog: np.ndarray = extract_hog_features(X_train, image_shape=face_shape)
    svc_model.fit(X_train_hog, y_train)

    # Optimize the model parameters
    svc_model = optimize_detector_parameters(svc_model, X_validation, y_validation)

    # Evaluate the model
    X_test_hog: np.ndarray = extract_hog_features(X_test, image_shape=face_shape)
    y_pred: np.ndarray = svc_model.predict(X_test_hog)
    print("SVC Model Evaluation:")
    print(classification_report(y_test, y_pred, target_names=["Non-Face", "Face"]))

    # Save the model
    save_detector(svc_model, face_shape, 'svc')

    print("Training KNN face detector with optimized parameters...")
    knn_model: Pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=100, whiten=True, random_state=42),
        KNeighborsClassifier(n_neighbors=7, weights='distance')
    )

    knn_model.fit(X_train_hog, y_train)
    y_pred = knn_model.predict(X_test_hog)
    print("KNN Model Evaluation:")
    print(classification_report(y_test, y_pred, target_names=["Non-Face", "Face"]))

    # Save the model
    save_detector(knn_model, face_shape, 'knn')

    # Test on the provided image
    image_path: str = "test_data" + os.sep + image

    try:
        svc_result: Optional[np.ndarray]
        svc_detections: List[Tuple[int, int, int, int, float]]
        svc_result, svc_detections = detect_faces(image_path, svc_model, face_shape, 'svc')
        cv2.imwrite("results/svc_detections.jpg", svc_result)

        knn_result: Optional[np.ndarray]
        knn_detections: List[Tuple[int, int, int, int, float]]
        knn_result, knn_detections = detect_faces(image_path, knn_model, face_shape, 'knn')
        cv2.imwrite("results/knn_detections.jpg", knn_result)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(svc_result, cv2.COLOR_BGR2RGB))
        plt.title("SVC Face Detection")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(knn_result, cv2.COLOR_BGR2RGB))
        plt.title("KNN Face Detection")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig("results/detection_comparison.png")
        plt.show()
    except Exception as e:
        print(f"Error testing on image: {e}")
        print("You can still use the saved models with your own images later.")


if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists("results"):
        os.makedirs("results")

    main("test.jpg")