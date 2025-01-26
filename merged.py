import moviepy.editor as mp
import sys

def stack_videos_vertically(video1_path, video2_path):
    # Load the two videos
    video1 = mp.VideoFileClip(video1_path)
    video2 = mp.VideoFileClip(video2_path)
    
    # Ensure both videos have the same duration
    video2 = video2.subclip(0, video1.duration)
    
    # Scale the second video to the width of the first, keeping the aspect ratio the same
    video2_resized = video2.resize(width=video1.w)
    
    # Stack the two videos vertically
    final_video = mp.clips_array([[video1], [video2_resized]])
    
    # Generate the output file name
    output_filename = f"{video1_path.split('.')[0]}_plotresult.mp4"
    
    # Write the result to the output file
    final_video.write_videofile(output_filename, codec="libx264", fps=24)
    
    print(f"Video saved as {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merged.py <video1_path> <video2_path>")
    else:
        video1_path = sys.argv[1]
        video2_path = sys.argv[2]
        stack_videos_vertically(video1_path, video2_path)
