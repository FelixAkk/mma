import numpy 
import matplotlib.pyplot as plot
import cv2 as opencv
import histdiff
import temporaldiff

from pydub import AudioSegment
import wave
import time
import sys

"""
This function processes a file by extracting visual features. 
"""
def process_video(file,f_features):
	print("Processing video with feature function: " + f_features.__name__)
	data		= []
	index	= 0
	
	# Here we use opencv to capture the video and derive the frame_count
	# from it.
	capture		= opencv.VideoCapture(file)
	frame_count	= capture.get(opencv.cv.CV_CAP_PROP_FRAME_COUNT)
	sec_per_frame	= 1 / capture.get(opencv.cv.CV_CAP_PROP_FPS)
	
	feat 			= None
	frame_old		= None
	features_old	= []
	features_new	= []
	# Start extracting frames from video.
	while(index < frame_count):
		# read 1 frame of video
		# frame is a numpy.ndarray object
		features_old 	= features_new
		(success,frame) = capture.read()
		
		if frame_old != None:
			feat = f_features(frame, frame_old)
		
		# only store if we've actually found a feature	
		if feat is not None:
			data.append(feat)
		
		frame_old = frame
		sys.stdout.write('\r' + str(index) + '/' + str(frame_count))
		index += 1
	capture.release()
	print("\n")
	if not data:
		print("[!] Error Reading Video")
	return data #numpy.asarray(data)


def get_keyframes(video_filename, output_path):
	print("Keyframe detection for file '" + video_filename + "'")
	
	data = process_video("./../media/" + video_filename, temporaldiff.extract)
	print(data)
	
get_keyframes("video_07.ogv","/")
