import numpy 
import matplotlib.pyplot as plot
import medialab as medialab

from pydub import AudioSegment
import wave
import time

def get_keyframes(video_filename, output_path):
	print("Keyframe detection for file '" + video_filename + "'")
	
	data = medialab.process_video("./../media/" + video_filename, extract_keyframes)

get_keyframes("video_07.ogv","/")
