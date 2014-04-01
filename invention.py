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
def process_video(file,f_features,frame_begin,frame_end):
	data		= []
	index		= frame_begin
	
	# Here we use opencv to capture the video and derive the frame_count
	# from it.
	capture		= opencv.VideoCapture(file)
	frame_count	= capture.get(opencv.cv.CV_CAP_PROP_FRAME_COUNT)
	sec_per_frame	= 1 / capture.get(opencv.cv.CV_CAP_PROP_FPS)
	
	feat 			= None
	frame_old		= None
	features_old	= []
	features_new	= []
	
	if frame_end < 0:
		frame_end = frame_count
	
	# Start extracting frames from video.
	while(index < frame_end):
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



# Set frame_end to -1 to process all frames
def get_keyframes(video_filename, output_path, frame_begin, frame_end):
	print("Keyframe detection for file '" + video_filename + "'")
	
	data = process_video("./../media/" + video_filename, temporaldiff.extract, frame_begin, frame_end)
	print(data)
	
get_keyframes("video_10.ogv","/", 0, 1000)
