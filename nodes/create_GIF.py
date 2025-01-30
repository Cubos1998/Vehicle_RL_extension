#!/usr/bin/env python3
"""
create_gif.py

Script to create an animated GIF from a sequence of images in a directory.
"""

import os
import glob
import imageio

def create_gif(image_folder, output_path, fps=5):
    """
    Creates an animated GIF from all images in `image_folder`.

    :param image_folder: Directory containing the images to compile.
    :param output_path: Filepath for the output GIF.
    :param fps: Frames per second (the higher the faster the animation).
    """
    # Gather all image file paths ending with .png, .jpg, or .jpeg (change as needed)
    # Sorting ensures frames are in the right order if filenames are sequential.
    file_paths = sorted(
        glob.glob(os.path.join(image_folder, "*.png")) 
        + glob.glob(os.path.join(image_folder, "*.jpg"))
        + glob.glob(os.path.join(image_folder, "*.jpeg"))
    )

    if not file_paths:
        print(f"No images found in {image_folder}. Exiting.")
        return

    # Create the GIF writer
    with imageio.get_writer(output_path, mode='I', fps=fps) as writer:
        for filename in file_paths:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"GIF successfully saved to {output_path}")

if __name__ == "__main__":
    # Example usage:
    folder_with_images = "./saved_preprocessed_observations"  # Change to your folder
    output_gif_path = "output.gif"  # Output file name

    create_gif(folder_with_images, output_gif_path, fps=5)
