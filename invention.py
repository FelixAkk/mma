import numpy 
import matplotlib.pyplot as plot
import cv2 as opencv
import histdiff
import temporaldiff

from pydub import AudioSegment
import wave
import time
import sys
import os, glob

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
	while(index <= frame_end):
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

def get_cuts(file, frame_begin, frame_end):
	data = process_video(file, histdiff.extract, frame_begin, frame_end)
	data_sorted = sorted(data, reverse=True)
	cutoffIndex = int(round(len(data) * 0.05)) # 20% highest diff peaks
	threshold = data_sorted[cutoffIndex]
	cut_mask = data > threshold
	cuts = [] # Let's build an array with the frame numbers of the cuts
	# Gotta merge clusters of cuts on milisecond level. We chose 
	# 15 frames, or 500ms, assuming ~30ftps.
	# (because a cutrate of > 2 cuts/s is insane and barely watchable,
	# hence unlikely)
	merge_threshold = 15
	peak_start = None
	peak_end   = None
	for idx in range(0, len(data)):
		if cut_mask[idx]:
			if peak_start is None:
				peak_start = idx
			if peak_end is None:
				peak_end = idx
			else: # Check if this cut is right behind another
				if idx - peak_end < merge_threshold:
					 peak_end = idx # merge it in the peak region
				else: # A peak has ended, and we just take the middel of it
					peak = peak_start + int(round(( peak_end - peak_start)/2 ))
					
					cuts.append(peak)
					peak_end   = None
					peak_start = None

	return cuts
	
def get_frames_by_index(video_filename, indices):
	print("Substracting detected keyframes from: '" + video_filename + "'\n")
	
	capture	= opencv.VideoCapture(video_filename)
	
	index = 0
	frame_end = max(indices)

	frames = []
	
	fps = capture.get(opencv.cv.CV_CAP_PROP_FPS)
	
	while (index <= frame_end):
		(success,frame) = capture.read()
		
		if index in indices:
			time = index/fps
			
			frames.append((index,frame,time))
		
		index += 1
	
	return frames
	
# Set frame_end to -1 to process all frames
def get_keyframes(video_filename, output_path, frame_begin, frame_end):
	print("Keyframe detection for file '" + video_filename + "'")
	
	cuts = [frame_begin] + get_cuts(video_filename, frame_begin, frame_end) + [frame_end]
	# energy = get_audio_energy("./../media/" + video_filename)
	# energy = get_audio_energy("./../media/" + video_filename)
	
	print("Cuts detected: " + str(cuts) + "\n")
	
	keyframes = []
	min_scene_length = 50
	
	for i in range(len(cuts) - 1):
		# Discard short scenes
		if cuts[i+1] - cuts[i] >= min_scene_length:
			# Take middle frame of scene as keyframe
			keyframes.append((cuts[i] + cuts[i+1]) / 2)
	
	print("Keyframes detected: " + str(keyframes) + "\n")
	
	return get_frames_by_index(video_filename, keyframes)
	
def generate_keyframes(video_filename, output_path, frame_begin, frame_end):
	frames = get_keyframes(video_filename, output_path, frame_begin, frame_end)
	
	# Change directory to output path
	os.chdir(output_path)
	
	# delete old keyframes in outputfolder
	print("Removing old keyframes...\n")
	
	old_files = glob.glob("*.jpg")
	
	for file in old_files:
		os.unlink(file)
	
	# dump to image files
	print("Dumping found keyframes in: '" + output_path + "'")
	
	for (index, frame, time) in frames:
		keyframe_file = "keyframe_" + str(index) + "_" + str(round(time*100)/100) + "s.jpg"
		print(" - " + keyframe_file)
		opencv.imwrite(keyframe_file,frame)

#generate_keyframes("./../media/video_10.ogv","/home/ilva/multimedia-lab/output/", 0, 18*30)
generate_keyframes("./../media/video_10.ogv","./output/", 0, 18*30)
