import numpy 
import matplotlib.pyplot as plot
import cv2 as opencv
import histdiff
import temporaldiff
import audioenergy

from scikits.audiolab import wavread
from scipy.io.wavfile import read, write

from pydub import AudioSegment
import wave
import time
import sys
import os, glob

"""
This function processes a file by extracting visual features. 
"""
def process_video(file,f_features,second_begin,second_end):
	frame_begin = second_begin * 30
	frame_end   = second_end * 30
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

"""
This function processes a file by extracting audio features. 
"""
def process_audio(file, f_features, second_begin, second_end):
	rate, ampl = read(file)
	data = []
	
	# we extract the mfcc data from precomputed files
	mfcc_file 	= file[:len(file)-3]+"mfcc"
	f 			= open(mfcc_file)	
	mfcc_data	= f.readline().split()
	samples_per_window 	= 1024
	start_sample 		= second_begin * rate
	end_sample			= start_sample + samples_per_window
	# uncomment the line below if processing the whole file takes too long
	max_samples			= min(second_end * rate,len(ampl))
	
	while end_sample < max_samples:
		# Give it all available data ...
		feat = audioenergy.extract(ampl[start_sample:end_sample], mfcc_data, start_sample, samples_per_window, max_samples)

		data.append(feat)
		start_sample += samples_per_window
		end_sample = min(end_sample+samples_per_window, max_samples)
		
		# print progress
		if (end_sample * 100) / max_samples % 5 == 0:
			sys.stdout.write("\r" + str( (end_sample * 100) / max_samples ) + "%")
			sys.stdout.flush()
	
	print 
	f.close()
	return data
	
def get_histdiff_cuts(file, second_begin, second_end):
	data = process_video(file, histdiff.extract, second_begin, second_end)
	data_sorted = sorted(data, reverse=True)
	cutoffIndex = int(round(len(data) * 0.07))
	threshold = data_sorted[cutoffIndex]
	cut_mask = data > threshold
	cuts = [] # Let's build an array with the frame numbers of the cuts
	# Gotta merge clusters of cuts on milisecond level. We chose 
	# 10 frames, or 300ms, assuming ~10ftps.
	# (because a cutrate of > 3 cuts/s is insane and barely watchable,
	# hence unlikely)
	merge_threshold = 10
	peak_start = None
	peak_end   = None
	for idx in range(0, len(data)):
		if cut_mask[idx]:
			print(idx)
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
	print(cuts)
	return cuts
	
def get_audioenergy_cuts(file, second_begin, second_end):
	data = process_audio(file, audioenergy.extract, second_begin, second_end)
	data_sorted = sorted(data, reverse=True)
	cutoffIndex = int(round(len(data) * 0.07))
	threshold = data_sorted[cutoffIndex]
	cut_mask = data > threshold
	cuts = [] # Let's build an array with the frame numbers of the cuts
	# Let's build an array with the frame numbers of the cuts
	# Gotta merge clusters of cuts on milisecond level. We chose 
	# 10 frames, or 300ms, assuming ~10ftps.
	# (because a cutrate of > 3 cuts/s is insane and barely watchable,
	# hence unlikely)
	merge_threshold = 10
	peak_start = None
	peak_end   = None
	for idx in range(0, len(data)):
		if cut_mask[idx]:
			print(idx)
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
def get_keyframes(filename, output_path, frame_begin, frame_end):
	print("Keyframe detection for file '" + filename + "'")
	
	video_filename = filename + '.ogv'
	audio_filename = filename + '.wav'
	
	cuts = [frame_begin] + get_cuts(video_filename, frame_begin, frame_end) + [frame_end]
	
	print("Cuts detected (based on video): " + str(cuts) + "\n")
	
	keyframes = []
	min_scene_length = 50
	
	for i in range(len(cuts) - 1):
		# Discard short scenes (length below threshold)
		if cuts[i+1] - cuts[i] >= min_scene_length:
			# Take middle frame of scene as keyframe
			keyframes.append((cuts[i] + cuts[i+1]) / 2)
	
	print("Keyframes detected: " + str(keyframes) + "\n")
	
	return get_frames_by_index(video_filename, keyframes)
	
def generate_keyframes(filename, output_path, frame_begin, frame_end):
	frames = get_keyframes(filename, output_path, frame_begin, frame_end)
	
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
generate_keyframes("./../media/video_10","./output/", 0, 18)
