import imageio
import cv2
import numpy as np
from PIL import Image as pillow
import os


def convert_images_to_gif_video(
    image_paths: list, output_path: str, 
    output_type: str ="gif", fps: int = 10,
    size: tuple = None
):
    """
    Convert a list of images to either GIF or video format.

    Parameters:
    image_paths (list): List of paths to input images
    output_path (str): Path where the output file should be saved
    output_type (str): Either 'gif' or 'video'
    fps (int): Frames per second for the output
    size (tuple): Optional (width, height) to resize images. If None, uses first image's size

    Returns:
    str: Path to the output file
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Read and process images
    images = []
    for img_path in image_paths:
        # Read image
        img = pillow.open(img_path)

        # Convert RGBA to RGB if necessary
        if img.mode == "RGBA":
            bg = pillow.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg

        # Resize if size is specified
        if size:
            img = img.resize(size, pillow.Resampling.LANCZOS)
        elif not images:  # Use first image size as default
            size = img.size

        # Convert to numpy array for video
        if output_type == "video":
            img = np.array(img)
            if len(img.shape) == 2:  # Convert grayscale to RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # Convert RGBA to RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        images.append(img)

    if output_type.lower() == "gif":
        # Create GIF
        imageio.mimsave(output_path, images, fps=fps, loop=0)  # 0 means loop forever

    elif output_type.lower() == "video":
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, size)

        # Write frames
        for img in images:
            out.write(img)

        # Release video writer
        out.release()

    return output_path
