import sys
import subprocess
import os

def main():
    if len(sys.argv)<4:
        print("python %s [input_video] [output_folder] [framestep]" % sys.argv[0])
        return

    input_video = sys.argv[1]
    output_folder = sys.argv[2]
    step = sys.argv[3]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    subprocess.run([
        "ffmpeg", "-i", input_video, 
        "-start_number", "0", 
        "-b:v", "10000k",  
        "-vsync", "0", 
        "-an", "-y", 
        "-q:v", "16", 
        "-vf", "framestep=%s" % step, 
        "%s/frame_%%06d.jpg" % output_folder])











if __name__ == "__main__":
    main()