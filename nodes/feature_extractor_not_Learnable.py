import cv2
import numpy as np
import matplotlib.pyplot as plt

def road_perception_comparison(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for plotting

    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Canny Edge Detection
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)

    # Hough Line Transform for Lane Detection
    lines_image = image.copy()
    edges_for_hough = cv2.Canny(gray, threshold1=50, threshold2=150)
    lines = cv2.HoughLinesP(edges_for_hough, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Color Segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_road = np.array([0, 0, 50])  # Adjust these values for your specific road color
    upper_road = np.array([180, 255, 200])
    road_mask = cv2.inRange(hsv, lower_road, upper_road)
    segmented_road = cv2.bitwise_and(image, image, mask=road_mask)

    # Perspective Transformation (Bird's-Eye View)
    h, w = gray.shape
    src_points = np.float32([[w // 4, h * 3 // 4], [w * 3 // 4, h * 3 // 4], [w, h], [0, h]])
    dst_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    birdseye = cv2.warpPerspective(image, matrix, (w, h))

    # Plot the comparison
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0, 0].imshow(image)
    axs[0, 0].set_title("Original Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(edges, cmap="gray")
    axs[0, 1].set_title("Canny Edge Detection")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(lines_image)
    axs[0, 2].set_title("Lane Detection (Hough Transform)")
    axs[0, 2].axis("off")

    axs[1, 0].imshow(road_mask, cmap="gray")
    axs[1, 0].set_title("Color Segmentation")
    axs[1, 0].axis("off")

    axs[1, 1].imshow(segmented_road)
    axs[1, 1].set_title("Segmented Road")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(birdseye)
    axs[1, 2].set_title("Bird's-Eye View (Perspective Transform)")
    axs[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

# Specify the path to your road image
image_path = "/home/cubos98/catkin_ws/src/Vehicle/first_image_received.jpg"  # Replace with the path to your road image
road_perception_comparison(image_path)
