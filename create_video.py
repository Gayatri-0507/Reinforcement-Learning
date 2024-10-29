import cv2
import os

def create_video(frame_folder, output_file):
    images = [img for img in os.listdir(frame_folder) if img.endswith(".png")]
    images.sort()  # Ensure the frames are in order

    frame = cv2.imread(os.path.join(frame_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(frame_folder, image)))

    video.release()

if __name__ == "__main__":
    create_video('frames', 'simulation_output.mp4')
