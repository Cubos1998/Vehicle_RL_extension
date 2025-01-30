import numpy as np
import cv2
import matplotlib.pyplot as plt


def preprocess_image(observation):
    # Convert RGBA to RGB if necessary
    if observation.shape[2] == 4:
        observation = cv2.cvtColor(observation, cv2.COLOR_RGBA2RGB)

    # Convert RGB to YUV color space (NVIDIA network uses YUV)
    #observation = cv2.cvtColor(observation, cv2.COLOR_RGB2YUV)

    # Resize to (66, 200), which is typical for NVIDIA's network
    observation = cv2.resize(observation, (80, 60), interpolation=cv2.INTER_AREA)

    # Normalize the image (similar to NVIDIA's normalization)
    #observation = observation / 255.0  # Scale pixel values to [0, 1]

    return observation

def main():
    # Example usage of preprocess_image function
    # Load an example image
    image_path = './first_image_received.jpg'
    observation = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if observation is None:
        print(f"Failed to load image at {image_path}")
        return

    # Preprocess the image
    preprocessed_image = preprocess_image(observation)

    # Print the shape of the preprocessed image
    print(f"Preprocessed image shape: {preprocessed_image.shape}")
    save_path = './preprocessed_image.jpg'  # Save the preprocessed image
    cv2.imwrite(save_path, preprocessed_image)

    # Display using matplotlib
    plt.imshow(preprocessed_image)
    plt.title('Preprocessed Image (Converted to RGB)')
    plt.axis('off')
    plt.show()

    print('Shape:', preprocessed_image.shape)
    print('Min pixel value:', preprocessed_image.min())
    print('Max pixel value:', preprocessed_image.max())
    print('Mean pixel value:', preprocessed_image.mean())
    print('Standard deviation:', preprocessed_image.std())

    # Flatten the pixel values for each channel
    Y_channel = preprocessed_image[:,:,0].flatten()
    print(Y_channel.shape)
    U_channel = preprocessed_image[:,:,0].flatten()
    print(U_channel)
    V_channel = preprocessed_image[:,:,0].flatten()
    print(V_channel)

    # Plot histograms
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(Y_channel, bins=100, color='gray')
    plt.title('Y Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(U_channel, bins=100, color='blue')
    plt.title('U Channel Histogram')
    plt.xlabel('Pixel Value')

    plt.subplot(1, 3, 3)
    plt.hist(V_channel, bins=100, color='red')
    plt.title('V Channel Histogram')
    plt.xlabel('Pixel Value')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()