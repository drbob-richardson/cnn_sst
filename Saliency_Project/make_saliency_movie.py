import os
import cv2

# Define the directories containing saliency images
directories = ["saliency_plots", "saliency_plots2"]

# Output video file name
output_file = "saliency_movie.mp4"

# Initialize variables for the images and delay
images = []
fps = 1  # Frames per second, corresponding to 1 second per image

# Collect all images from both directories in order
for directory in directories:
    for i in range(1, 19):  # Assumes folder names are lead_1, lead_2, ..., lead_18
        subfolder = os.path.join(directory, f"lead_{i}")
        image_path = os.path.join(subfolder, f"saliency_lead_{i}.png")
        if os.path.exists(image_path):
            images.append(image_path)

# Load images and get dimensions
loaded_images = [cv2.imread(image) for image in images if os.path.exists(image)]
height, width, layers = loaded_images[0].shape

# Define the video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Add each image to the video
for img in loaded_images:
    video.write(img)

# Release the video writer
video.release()

print(f"Video saved as {output_file}")
