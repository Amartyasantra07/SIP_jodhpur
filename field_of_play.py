import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def show_images(images, titles, cols=3, cmap='gray'):
    assert(len(images) == len(titles))
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def segment_field_of_play(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded properly
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for green color in HSV
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green color
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of the green area
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the field of play
    field_mask = np.zeros_like(green_mask)
    if contours:
        # Assume the largest contour corresponds to the field of play
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(field_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    else:
        print("Warning: No contours found, returning original image.")
        return image_rgb

    # Mask the original image with the field mask
    field_segmented = cv2.bitwise_and(image_rgb, image_rgb, mask=field_mask)

    # Save the segmented field image
    segmented_image_path = os.path.join('output', 'segmented_field.png')
    os.makedirs('output', exist_ok=True)
    cv2.imwrite(segmented_image_path, cv2.cvtColor(field_segmented, cv2.COLOR_RGB2BGR))
    print(f"Segmented field image saved at {segmented_image_path}")

    # Display the segmented field image
    images = [field_segmented]
    titles = ['Field of Play Segmented']
    
    show_images(images, titles, cols=1, cmap=None)
    
    return field_segmented

def detect_lines(image_path):
    # Segment the field of play
    image = segment_field_of_play(image_path)
    
    # Check if the segmentation was successful
    if image is None:
        print("Error: Segmentation failed, cannot detect lines.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detector to find edges
    edges = cv2.Canny(blurred, 50, 150)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    # Create an image to draw lines on
    line_image = np.copy(image)

    # Draw the detected lines on the image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the image with detected lines
    lines_image_path = os.path.join('output', 'detected_lines.png')
    cv2.imwrite(lines_image_path, cv2.cvtColor(line_image, cv2.COLOR_RGB2BGR))
    print(f"Detected lines image saved at {lines_image_path}")

    # Display the original image, edges, and line image
    images = [image, edges, line_image]
    titles = ['Original Image', 'Edges', 'Detected Lines']
    
    show_images(images, titles, cols=3, cmap='gray')

# Example usage
image_path = 'image_dataset/Images_ground/m01_s05.png'
detect_lines(image_path)
