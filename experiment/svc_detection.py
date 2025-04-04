import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_lfw_people
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
from skimage.feature import hog
from typing import Tuple, List, Optional, Dict, Any, Union
import joblib, os
from skimage import data
from skimage.transform import resize
from scipy import ndimage


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
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        for _ in range(n_samples // len(sample_images) + 1):
            if img.shape[0] > h and img.shape[1] > w:
                i = np.random.randint(0, img.shape[0] - h)
                j = np.random.randint(0, img.shape[1] - w)
                patch = img[i:i + h, j:j + w]
                non_face_images.append(patch)
            else:
                resized = resize(img, (h, w), anti_aliasing=True)
                if isinstance(resized, np.ndarray) and resized.ndim == 2:
                    non_face_images.append(resized)

    # Removed the digits dataset as requested
    print("Generating more texture patterns as additional non-face examples...")

    # Checkerboard patterns with varying scales
    for scale in range(3, 30, 3):
        checkerboard = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                if (i // scale + j // scale) % 2 == 0:
                    checkerboard[i, j] = 1
        non_face_images.append(checkerboard)

    # Stripes patterns with varying orientations and scales
    for scale in range(3, 30, 3):
        # Horizontal stripes
        stripes_h = np.zeros((h, w))
        for i in range(h):
            if (i // scale) % 2 == 0:
                stripes_h[i, :] = 1
        non_face_images.append(stripes_h)

        # Vertical stripes
        stripes_v = np.zeros((h, w))
        for j in range(w):
            if (j // scale) % 2 == 0:
                stripes_v[:, j] = 1
        non_face_images.append(stripes_v)

        # Diagonal stripes
        stripes_d = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                if ((i + j) // scale) % 2 == 0:
                    stripes_d[i, j] = 1
        non_face_images.append(stripes_d)

    # Random noise patterns with different distributions
    for _ in range(n_samples // 5):
        # Uniform noise
        uniform_noise = np.random.rand(h, w)
        non_face_images.append(uniform_noise)

        # Gaussian noise
        gaussian_noise = np.random.normal(0.5, 0.2, (h, w))
        gaussian_noise = np.clip(gaussian_noise, 0, 1)
        non_face_images.append(gaussian_noise)

        # Salt and pepper noise
        salt_pepper = np.zeros((h, w))
        salt_pepper[np.random.rand(h, w) < 0.1] = 1
        non_face_images.append(salt_pepper)

    # Gradient patterns
    # Horizontal gradient
    h_gradient = np.linspace(0, 1, w)
    h_gradient = np.tile(h_gradient, (h, 1))
    non_face_images.append(h_gradient)

    # Vertical gradient
    v_gradient = np.linspace(0, 1, h).reshape(-1, 1)
    v_gradient = np.tile(v_gradient, (1, w))
    non_face_images.append(v_gradient)

    # Radial gradient
    y, x = np.ogrid[-h / 2:h / 2, -w / 2:w / 2]
    r_gradient = np.sqrt(x * x + y * y)
    r_gradient = r_gradient / np.max(r_gradient)
    non_face_images.append(r_gradient)

    # Geometric shapes
    for shape_type in range(3):
        shape_img = np.zeros((h, w))

        if shape_type == 0:  # Circle
            rr, cc = np.ogrid[:h, :w]
            center_r, center_c = h // 2, w // 2
            radius = min(h, w) // 3
            circle_mask = (rr - center_r) ** 2 + (cc - center_c) ** 2 <= radius ** 2
            shape_img[circle_mask] = 1

        elif shape_type == 1:  # Rectangle
            margin = min(h, w) // 4
            shape_img[margin:h - margin, margin:w - margin] = 1

        elif shape_type == 2:  # Triangle
            points = np.array([[h // 2, 0], [0, h - 1], [w - 1, h - 1]])
            from skimage.draw import polygon
            rr, cc = polygon(points[:, 0], points[:, 1], (h, w))
            shape_img[rr, cc] = 1

        non_face_images.append(shape_img)

    # Wave patterns
    for freq in [3, 5, 10, 20]:
        # Sine wave pattern
        x = np.linspace(0, 2 * np.pi, w)
        y = np.linspace(0, 2 * np.pi, h).reshape(-1, 1)
        sin_pattern = np.sin(freq * x) * np.sin(freq * y)
        sin_pattern = (sin_pattern + 1) / 2  # Normalize to [0, 1]
        non_face_images.append(sin_pattern)

    # Filtered noise (different textures)
    for sigma in [1, 3, 5]:
        blurred_noise = ndimage.gaussian_filter(np.random.rand(h, w), sigma=sigma)
        non_face_images.append(blurred_noise)

    print("Adding crops from LFW images that don't contain facial features...")
    for _ in range(n_samples // 2):
        random_idx = np.random.randint(0, len(face_images))
        img = face_images[random_idx]

        corners = [
            (0, 0),
            (0, w - h // 3),
            (h - h // 3, 0),
            (h - h // 3, w - w // 3)
        ]

        corner_idx = np.random.randint(0, len(corners))
        y, x = corners[corner_idx]
        crop_h = h // 3
        crop_w = w // 3

        if y + crop_h <= h and x + crop_w <= w:
            crop = img[y:y + crop_h, x:x + crop_w]
            crop_resized = cv2.resize(crop, (w, h))
            non_face_images.append(crop_resized)

    non_face_images_array = np.array(non_face_images)

    for i in range(len(non_face_images_array)):
        if non_face_images_array[i].max() > 1.0:
            non_face_images_array[i] = non_face_images_array[i] / 255.0

    non_face_labels = np.zeros(len(non_face_images_array))

    print(f"Created {len(non_face_images_array)} non-face examples")

    min_samples = min(len(face_images), len(non_face_images_array))
    face_indices = np.random.choice(len(face_images), min_samples, replace=False)
    non_face_indices = np.random.choice(len(non_face_images_array), min_samples, replace=False)

    face_subset = face_images[face_indices]
    non_face_subset = non_face_images_array[non_face_indices]
    face_labels_subset = face_labels[face_indices]
    non_face_labels_subset = non_face_labels[non_face_indices]

    X = np.vstack([
        face_subset.reshape(len(face_subset), -1),
        non_face_subset.reshape(len(non_face_subset), -1)
    ])
    y = np.hstack([face_labels_subset, non_face_labels_subset])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of face examples: {np.sum(y == 1)}")
    print(f"Number of non-face examples: {np.sum(y == 0)}")

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


def create_selective_search_detection(image: np.ndarray, max_boxes: int = 1000) -> List[Tuple[int, int, int, int]]:
    """
    Uses Selective Search algorithm to generate object proposal boxes for face detection.

    Selective Search is a hierarchical grouping algorithm that generates
    region proposals based on color, texture, size, and shape compatibility.

    Args:
        image (np.ndarray): Input image
        max_boxes (int): Maximum number of boxes to generate

    Returns:
        List[Tuple[int, int, int, int]]: List of object proposal boxes as (x, y, w, h)
    """
    # If image is grayscale, convert to BGR for Selective Search
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    # Create Selective Search segmentation
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # If cv2.ximgproc is not available, use a fallback method
    if not hasattr(cv2, 'ximgproc'):
        return create_fallback_region_proposals(image, max_boxes)

    # Initialize with the input image
    ss.setBaseImage(image_color)

    # Use fast strategy which is a good balance between quality and speed
    ss.switchToSelectiveSearchFast()

    # Get region proposals
    rects = ss.process()

    # Convert to our format (x, y, w, h)
    proposals = []

    # Filter by size and aspect ratio for face-like regions
    min_width, min_height = 20, 20
    for x, y, w, h in rects[:max_boxes]:
        # Skip very small regions
        if w < min_width or h < min_height:
            continue

        # Calculate aspect ratio (width/height)
        aspect_ratio = w / float(h)

        # Face aspect ratios typically fall between 0.5 and 1.5
        if 0.5 <= aspect_ratio <= 1.5:
            proposals.append((x, y, w, h))

    # Limit number of proposals
    return proposals[:max_boxes]


def create_fallback_region_proposals(image: np.ndarray, max_boxes: int = 1000) -> List[Tuple[int, int, int, int]]:
    """
    Fallback function when Selective Search is not available.
    Uses a combination of sliding windows and edge-based detection.

    Args:
        image (np.ndarray): Input image
        max_boxes (int): Maximum number of boxes to generate

    Returns:
        List[Tuple[int, int, int, int]]: List of object proposal boxes as (x, y, w, h)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    height, width = gray.shape
    proposals = []

    # Step 1: Use edge detection for region proposals
    edges = cv2.Canny(gray, 50, 150)

    # Apply dilation to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours from the edges
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours to bounding boxes
    min_width, min_height = 20, 20  # Minimum box dimensions

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out very small boxes
        if w > min_width and h > min_height:
            # Calculate aspect ratio (width/height)
            aspect_ratio = w / float(h)

            # Face aspect ratios typically fall between 0.5 and 1.5
            if 0.5 <= aspect_ratio <= 1.5:
                proposals.append((x, y, w, h))

    # Step 2: Add strategic sliding windows with different aspect ratios
    # This helps ensure coverage in areas where edge detection might miss faces
    window_sizes = [
        (int(width * 0.1), int(height * 0.1)),  # Small
        (int(width * 0.2), int(height * 0.2)),  # Medium
        (int(width * 0.3), int(height * 0.3)),  # Large
    ]

    # Determine step size based on image size (smaller steps for smaller images)
    step_ratio = 0.25  # 25% overlap between windows

    for win_w, win_h in window_sizes:
        step_x = max(1, int(win_w * step_ratio))
        step_y = max(1, int(win_h * step_ratio))

        for y in range(0, height - win_h + 1, step_y):
            for x in range(0, width - win_w + 1, step_x):
                proposals.append((x, y, win_w, win_h))

    # Prioritize proposals by size (with preference for mid-sized regions)
    def score_proposal(box):
        x, y, w, h = box
        area = w * h
        ideal_area = width * height * 0.1  # Roughly 10% of image is ideal for a face
        return -abs(area - ideal_area)  # Negative because we want to maximize

    proposals.sort(key=score_proposal)

    # Limit number of proposals
    return proposals[:max_boxes]


def create_image_pyramid(image: np.ndarray, min_size: int = 30,
                         scale_factor: float = 1.2) -> List[Tuple[np.ndarray, float]]:
    """
    Creates an image pyramid by progressively scaling down the image.

    This allows detection of faces at multiple scales without resizing the detection window.

    Args:
        image (np.ndarray): Input image
        min_size (int): Minimum size of the smallest layer in the pyramid
        scale_factor (float): How much to scale down each layer

    Returns:
        List[Tuple[np.ndarray, float]]: List of (scaled_image, scale_factor) pairs
    """
    # Initialize the pyramid
    pyramid = []

    # Add the original image
    pyramid.append((image.copy(), 1.0))

    # Calculate the current scale
    current_scale = 1.0
    current_image = image.copy()

    # Keep scaling down until we reach the minimum size
    while True:
        # Calculate the new dimensions
        current_scale *= scale_factor
        new_width = int(image.shape[1] / current_scale)
        new_height = int(image.shape[0] / current_scale)

        # If either dimension becomes too small, break
        if new_width < min_size or new_height < min_size:
            break

        # Resize the image
        resized = cv2.resize(image, (new_width, new_height))

        # Add to pyramid
        pyramid.append((resized, current_scale))

    return pyramid


def detect_faces_with_selective_search(image: np.ndarray, model: Pipeline,
                                       face_shape: Tuple[int, int],
                                       min_confidence: float = 0.7) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect faces in an image using Selective Search for region proposals and a classifier
    to determine if each proposal contains a face.

    Args:
        image (np.ndarray): Input image
        model (Pipeline): Trained classifier model
        face_shape (Tuple[int, int]): Shape of face images used in training (height, width)
        min_confidence (float): Minimum confidence threshold

    Returns:
        List[Tuple[int, int, int, int, float]]: List of face detections (x, y, w, h, confidence)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = face_shape
    detections = []

    # Create image pyramid
    pyramid = create_image_pyramid(gray)

    for layer, scale in pyramid:
        # Get object proposals using Selective Search
        proposals = create_selective_search_detection(layer)

        if not proposals:
            continue

        # Process each proposal
        windows = []
        coords = []

        for (x, y, box_w, box_h) in proposals:
            # Ensure the proposal is of a reasonable size to contain a face
            if box_w < 10 or box_h < 10:
                continue

            # Extract the region
            roi = layer[y:y + box_h, x:x + box_w]

            # Skip if ROI is empty or has wrong dimensions
            if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
                continue

            # Resize to match the face shape expected by the model
            roi_resized = cv2.resize(roi, (w, h))

            # Normalize if needed
            if roi_resized.max() > 1.0:
                roi_resized = roi_resized / 255.0

            windows.append(roi_resized)
            coords.append((x, y, box_w, box_h, scale))

        if not windows:
            continue

        # Extract HOG features for all windows
        windows_hog = extract_hog_features(windows, image_shape=face_shape)

        # Get predictions from the model
        if hasattr(model, 'predict_proba'):
            confidences = model.predict_proba(windows_hog)[:, 1]
        else:
            predictions = model.predict(windows_hog)
            confidences = np.array([1.0 if p == 1 else 0.0 for p in predictions])

        # Add detections that meet the confidence threshold
        for (x, y, box_w, box_h, s), conf in zip(coords, confidences):
            if conf >= min_confidence:
                # Convert back to original image coordinates
                orig_x = int(x * s)
                orig_y = int(y * s)
                orig_w = int(box_w * s)
                orig_h = int(box_h * s)

                detections.append((orig_x, orig_y, orig_w, orig_h, conf))

    # Apply non-maximum suppression to remove overlapping detections
    detections = non_max_suppression(detections, overlap_threshold=0.3)

    return detections


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

    boxes: np.ndarray = np.array([d[:4] for d in detections])
    scores: np.ndarray = np.array([d[4] for d in detections])

    x1: np.ndarray = boxes[:, 0]
    y1: np.ndarray = boxes[:, 1]
    x2: np.ndarray = boxes[:, 0] + boxes[:, 2]
    y2: np.ndarray = boxes[:, 1] + boxes[:, 3]

    area: np.ndarray = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs: np.ndarray = np.argsort(scores)[::-1]

    pick: List[int] = []
    while len(idxs) > 0:
        i: int = idxs[0]
        pick.append(i)

        idxs = idxs[1:]
        if len(idxs) == 0:
            break

        xx1: np.ndarray = np.maximum(x1[i], x1[idxs])
        yy1: np.ndarray = np.maximum(y1[i], y1[idxs])
        xx2: np.ndarray = np.minimum(x2[i], x2[idxs])
        yy2: np.ndarray = np.minimum(y2[i], y2[idxs])

        w: np.ndarray = np.maximum(0, xx2 - xx1 + 1)
        h: np.ndarray = np.maximum(0, yy2 - yy1 + 1)
        intersection: np.ndarray = w * h

        iou: np.ndarray = intersection / (area[i] + area[idxs] - intersection)

        idxs = np.delete(idxs, np.where(iou > overlap_threshold)[0])

    return [detections[i] for i in pick]


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
    if 'svc' in model.named_steps:
        svm = model.named_steps['svc']
        svm.class_weight = {0: 1.5, 1: 1.0}

    return model


def detect_faces(image_path: str, model: Pipeline,
                 face_shape: Tuple[int, int]) -> Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int, float]]]:
    """
    Detect faces in an image using SVC detector with Selective Search and image pyramid.

    Args:
        image_path (str): Path to the input image
        model (Pipeline): Trained classifier model
        face_shape (Tuple[int, int]): Shape of face images used in training (height, width)

    Returns:
        Tuple[Optional[np.ndarray], List[Tuple[int, int, int, int, float]]]:
            - Image with visualized detections (or None if image cannot be loaded)
            - List of detections, each as (x, y, width, height, confidence)
    """
    image: Optional[np.ndarray] = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image at {image_path}")
        return None, []

    gray: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Selective Search with image pyramid for detection
    detections = detect_faces_with_selective_search(
        gray,
        model,
        face_shape,
        min_confidence=0.7
    )

    print(f"Found {len(detections)} faces using SVC detector with Selective Search")

    # Draw detections on the image
    result_img: np.ndarray = image.copy()
    for (x, y, w, h, confidence) in detections:
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result_img, f"{confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return result_img, detections


def save_detector(model: Pipeline, face_shape: Tuple[int, int]) -> None:
    """
    Save the trained detector to disk.

    This function saves the trained model and its metadata (face shape)
    to files that can be loaded later.

    Args:
        model (Pipeline): Trained classifier pipeline
        face_shape (Tuple[int, int]): Height and width of the face images used for training
    """
    filename: str = "models/svc_face_detector.joblib"
    joblib.dump(model, filename)

    with open("models/detector_metadata.txt", "w") as f:
        f.write(f"Face shape: {face_shape[0]}x{face_shape[1]}\n")
        f.write("Detector type: svc\n")

    print(f"Detector saved as {filename}")


def load_detector() -> Tuple[Pipeline, Tuple[int, int]]:
    """
    Load a saved detector from disk.

    This function loads a trained model and its metadata from files.

    Returns:
        Tuple[Pipeline, Tuple[int, int]]:
            - Loaded model
            - Face shape (height, width)
    """
    filename: str = "models/svc_face_detector.joblib"
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model file {filename} not found. Please train the model first.")

    model: Pipeline = joblib.load(filename)

    metadata_file = "models/detector_metadata.txt"
    if not os.path.exists(metadata_file):
        # Default face shape if metadata is missing
        print("Warning: Metadata file not found. Using default face shape.")
        return model, (62, 47)  # Default LFW shape

    with open(metadata_file, "r") as f:
        lines: List[str] = f.readlines()
        face_shape_str: str = lines[0].split(":")[1].strip()
        h, w = [int(x) for x in face_shape_str.split("x")]

    return model, (h, w)


def main(image: str) -> None:
    """
    Main function to train and test SVC face detector.

    This function:
    1. Creates a training dataset of face and non-face examples from diverse sources
    2. Trains SVC face detector with optimized parameters
    3. Evaluates the detector on test data
    4. Saves the trained detector
    5. Tests it on a new image if available
    """
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

    X_test, X_validation, y_test, y_validation = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42)

    print("Training SVC face detector with optimized parameters...")

    # Convert to HOG features for training
    X_train_hog: np.ndarray = extract_hog_features(X_train, image_shape=face_shape)
    X_test_hog: np.ndarray = extract_hog_features(X_test, image_shape=face_shape)
    X_validation_hog: np.ndarray = extract_hog_features(X_validation, image_shape=face_shape)

    # Create and train SVC model
    svc_model: Pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=100, whiten=True, random_state=42),
        SVC(kernel='linear', probability=True, class_weight='balanced', C=10.0)
    )
    svc_model.fit(X_train_hog, y_train)

    # Optimize parameters for the model
    svc_model = optimize_detector_parameters(svc_model, X_validation_hog, y_validation)

    # Evaluate SVC model
    y_pred_svc: np.ndarray = svc_model.predict(X_test_hog)
    print("SVC Model Evaluation:")
    print(classification_report(y_test, y_pred_svc, target_names=["Non-Face", "Face"]))

    # Save the trained model
    save_detector(svc_model, face_shape)

    # Test on a new image if provided
    if image:
        image_path: str = "test_data" + os.sep + image
        try:
            result_img: Optional[np.ndarray]
            detections: List[Tuple[int, int, int, int, float]]
            result_img, detections = detect_faces(image_path, svc_model, face_shape)

            if result_img is not None:
                cv2.imwrite("results/face_detections.jpg", result_img)
                print(f"Detection results saved to results/face_detections.jpg")
        except Exception as e:
            print(f"Error testing on image: {e}")
            print("You can still use the saved model with your own images later.")


if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")

    if not os.path.exists("results"):
        os.makedirs("results")

    main("test.jpg")