import pixellib
from IPython.display import Video
from tensorflow.keras.models import Sequential
from pixellib.semantic import semantic_segmentation
import cv2

# Setting the file path for the input video
input_video_file_path = "C:\\Users\\Gulsu\\Desktop\\Earthquake Classroom Video.mp4"

# Displaying the input video
Video(input_video_file_path, embed=True)

# Creating a semantic_segmentation object
segment_video = semantic_segmentation()

# Loading the Xception model trained on ade20k dataset
segment_video.load_ade20k_model("deeplabv3_xception65_ade20k.h5")

# Processing the video
segment_video.process_video_ade20k(
    input_video_file_path,
    frames_per_second=30,
    output_video_name="C:\\Users\\Gulsu\\Desktop\\semantic_segmentation_output.mp4")
