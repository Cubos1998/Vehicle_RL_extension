import cv2
import numpy as np
import matplotlib.pyplot as plt

def track_dashed_yellow_lanes(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

    # Convert to HSV for yellow detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define yellow color range
    lower_yellow = np.array([20, 100, 100])  # Adjust these values for yellow detection
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for yellow lanes
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Morphological operations to bridge gaps in dashed lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    yellow_mask_dilated = cv2.dilate(yellow_mask, kernel, iterations=2)

    # Canny Edge Detection
    edges = cv2.Canny(yellow_mask_dilated, threshold1=50, threshold2=150)

    # Hough Line Transform to detect lane lines
    lines_image = image.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=50)
    
    if lines is not None:
        slopes = []
        intercepts = []
        
        # Group lines into a single lane
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
            intercept = y1 - slope * x1
            slopes.append(slope)
            intercepts.append(intercept)

            # Draw the detected line segments
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Calculate average slope and intercept for lane extrapolation
        if slopes:
            avg_slope = np.mean(slopes)
            avg_intercept = np.mean(intercepts)

            # Extrapolate lane line
            h, w, _ = image.shape
            y1, y2 = h, int(h * 0.6)
            x1 = int((y1 - avg_intercept) / avg_slope)
            x2 = int((y2 - avg_intercept) / avg_slope)
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    # Plot the results
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(yellow_mask_dilated, cmap="gray")
    axs[1].set_title("Yellow Mask (Bridged)")
    axs[1].axis("off")

    axs[2].imshow(lines_image)
    axs[2].set_title("Tracked Dashed Yellow Lanes")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()

# Specify the path to your road image
image_path = "/home/cubos98/catkin_ws/src/Vehicle/first_image_received.jpg"  # Replace with your image path
track_dashed_yellow_lanes(image_path)
