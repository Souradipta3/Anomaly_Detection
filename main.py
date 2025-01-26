from sys import argv
from os import system

try:
    filepath = argv[1]
except:
    print("Usage: python main.py <video_path>")
    exit(1)
    
system(f"python video2frame.py --n {filepath}")
system(f"python vis.py --n {filepath[:-4]}")
system(f"python merged.py {filepath[:-4]}_result.mp4 {filepath[:-4]}_result_plot.mp4")

