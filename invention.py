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
	fps = capture.get(opencv.cv.CV_CAP_PROP_FPS)
	sec_per_frame	= 1 / fps
	
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
	return (data, fps) #numpy.asarray(data)

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
	samples_per_window 	= rate/29.97 #1024
	start_sample 		= second_begin * rate
	end_sample			= start_sample + samples_per_window
	# uncomment the line below if processing the whole file takes too long
	max_samples			= min((second_end+1) * rate, len(ampl))
	
	while end_sample < max_samples:
		# Give it all available data ...
		feat = f_features(ampl[start_sample:end_sample], mfcc_data, start_sample, samples_per_window, max_samples)

		data.append(feat)
		start_sample += samples_per_window
		end_sample = min(end_sample+samples_per_window, max_samples)
		
		# print progress
		if (end_sample * 100) / max_samples % 5 == 0:
			sys.stdout.write("\r" + str( (end_sample * 100) / max_samples ) + "%")
			sys.stdout.flush()
	
	f.close()
	return (data, rate)
	
def get_histdiff_cuts(file, second_begin, second_end):
	(data, fps) = process_video(file, histdiff.extract, second_begin, second_end)
	data_sorted = sorted(data, reverse=True)
	cutoffIndex = int(round(len(data) * 0.2))
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
	return [second_begin*30] + cuts + [second_end*30]
	
def get_audioenergy_peaks(file, second_begin, second_end, cuts):
	(data, rate) = process_audio(file, audioenergy.extract, second_begin, second_end)
	peaks = []
	window_length = 1024
	
	min_scene_length = 50
	for i in range(len(cuts) - 1):
		# Discard short scenes (length below threshold)
		if cuts[i+1] - cuts[i] <= min_scene_length:
			print("to short cut, ignoring" + str(cuts[i]) + " to " + str(cuts[i+1]))
			continue
		peak = -1
		peakIdx = -1
		for idx in range(cuts[i], cuts[i+1]):
			if data[idx] > peak:
				peak = data[idx]
				peakIdx = idx
		peaks.append(peakIdx)
	return peaks

	
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
def get_keyframes(filename, output_path, second_begin, second_end):
	print("Keyframe detection for file '" + filename + "'")
	
	video_filename = filename + '.ogv'
	audio_filename = filename + '.wav'
	
	cuts = get_histdiff_cuts(video_filename, second_begin, second_end)
	print("Cuts detected (based on video): " + str(cuts) + "\n")
	peaks = get_audioenergy_peaks(audio_filename, second_begin, second_end, cuts)
	print("Peaks detected (based on audio): " + str(peaks) + "\n")
	
	#keyframes = []
	#print("Keyframes detected: " + str(keyframes) + "\n")
	
	return get_frames_by_index(video_filename, peaks)
	
def generate_keyframes(filename, output_path, second_begin, second_end):
	frames = get_keyframes(filename, output_path, second_begin, second_end)
	
	total_histdiff = 0
	i = 0
	
	histdiffs = []
	
	for (index, frame, time) in frames:
		for (index_other, frame_other, time_other) in frames:
			if index != index_other:
				i += 1
				cur_histdiff = histdiff.extract(frame, frame_other)
				total_histdiff += cur_histdiff
				
				histdiffs.append((index, index_other, cur_histdiff))
				
				print "a: " + str(index) + " b: " + str(index_other) + "  " + str(cur_histdiff)
				
	avg_histdiff = total_histdiff / i
				
	print "Average histdiff between keyframes: " + str(avg_histdiff) + "\n"
	
	duplicates = []
	
	for (index, index_other, diff) in histdiffs:
		if diff < avg_histdiff * 0.5:
			duplicates.append(index_other)
	
	print "Duplicates found: " + str(duplicates)
	
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
		if not index in duplicates:
			keyframe_file = "keyframe_" + str(index) + "_" + str(round(time*100)/100) + "s.jpg"
			print(" - " + keyframe_file)
			opencv.imwrite(keyframe_file,frame)

#generate_keyframes("./../media/video_10.ogv","/home/ilva/multimedia-lab/output/", 0, 18*30)
generate_keyframes("./../media/video_10","./output/", 0, 120)
