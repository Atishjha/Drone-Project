import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def analyze_surface_laplacian(image_path):
    """
    Analyze surface roughness using Laplacian edge detection.
    Parameters:
        image_path (str): Path to the input image.
    """
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load the image.")
        return

    # Resize the image for faster processing (optional)
    image = cv2.resize(image, (500, 500))

    # Apply a Laplacian filter to detect edges (highlights roughness)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Compute the variance of the Laplacian (higher = rough, lower = smooth)
    variance = np.var(laplacian)

    # Set a threshold to distinguish smooth vs. rough surfaces
    threshold = 100  # Adjust this value based on testing and image resolution

    # Analyze and print the result
    if variance < threshold:
        result = "Smooth Surface"
    else:
        result = "Rough Surface"

    print(f"[Laplacian] {result} (Variance: {variance:.2f})")

    # Display the original and edge-detected images
    cv2.imshow("Original Image", image)
    cv2.imshow("Edge Detection", np.uint8(np.absolute(laplacian)))  # Convert to uint8 for display
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def analyze_surface_texture(image_path):
    """
    Analyze surface roughness using Texture Analysis (GLCM).
    Parameters:
        image_path (str): Path to the input image.
    """
    # Load the image and convert to grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not load the image.")
        return

    # Resize the image for faster processing (optional)
    image = cv2.resize(image, (500, 500))

    # Normalize the image to have values between 0 and 255 (if required)
    normalized_img = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Compute the GLCM (Gray Level Co-occurrence Matrix)
    glcm = graycomatrix(normalized_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    # Extract texture features from the GLCM
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Print texture features
    print(f"[GLCM] Contrast: {contrast:.2f}")
    print(f"[GLCM] Homogeneity: {homogeneity:.2f}")
    print(f"[GLCM] Energy: {energy:.2f}")
    print(f"[GLCM] Correlation: {correlation:.2f}")

    # Define thresholds for rough vs. smooth classification (adjust as needed)
    if contrast < 5 and energy > 0.3 and homogeneity > 0.5:
        result = "Smooth Surface"
    else:
        result = "Rough Surface"

    print(f"[GLCM] {result}")

    # Display the original image with the result
    cv2.putText(image, f"{result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "C:/Users/KIIT0001/Downloads/surface_image.jpg"  # Replace with your image file path

    # Call both methods for surface analysis
    print("Analyzing surface using Laplacian method...")
    analyze_surface_laplacian(image_path)

    print("\nAnalyzing surface using Texture Analysis (GLCM)...")
    analyze_surface_texture(image_path)
