import glob
import sys
import pickle
import numpy 
import cv2 as opencv
import matplotlib.pyplot as plot
import re as regex
import math
from pydub import AudioSegment
import tempfile
import wave
import os
import struct
import scipy
from scikits.audiolab import wavread
import time
from scipy.io.wavfile import read, write


"""
This function extracts features from a given frame. It extracts a normalized histogram for each of the color channels
of the frame. 
"""
def extract_histogram(frame, prev_frame):
	shape = frame.shape
	num_pixels = shape[0]*shape[1]
	# Write a function that returns a histogram given a NxMx1 frame. 
	# Hint: You don't have to reinvent the wheel. Also consider normalization
	hist_feature_r = []
	hist_feature_g = []
	hist_feature_b = []

	mask = numpy.ones(frame[:,:,0].shape).astype('uint8')
	
	red =   [frame[:,:,0].astype('uint8')]
	green = [frame[:,:,1].astype('uint8')]
	blue =  [frame[:,:,2].astype('uint8')]
	prev_red =   [prev_frame[:,:,0].astype('uint8')]
	prev_green = [prev_frame[:,:,1].astype('uint8')]
	prev_blue =  [prev_frame[:,:,2].astype('uint8')]
	hist_feature_r = numpy.linalg.norm(opencv.calcHist(red,   [0], mask, [256], [0,255]) - opencv.calcHist(prev_red,   [0], mask, [256], [0,255]))
	hist_feature_g = numpy.linalg.norm(opencv.calcHist(green, [0], mask, [256], [0,255]) - opencv.calcHist(prev_green, [0], mask, [256], [0,255]))
	hist_feature_b = numpy.linalg.norm(opencv.calcHist(blue,  [0], mask, [256], [0,255]) - opencv.calcHist(prev_blue,  [0], mask, [256], [0,255]))
	
	#features = numpy.concatenate( (hist_feature_r[:], hist_feature_g[:], hist_feature_b[:]) )
	#return features
	return numpy.sum(hist_feature_r) + numpy.sum(hist_feature_g) + numpy.sum(hist_feature_b)
	
def extract_temporal_difference(frame, prev_frame):
	# implement the sum of pixel differences between two frames
	shape 		= frame.shape
	num_pixels 	= shape[0]*shape[1]
	#diff = abs(frame - prev_frame)
	#sum = 0
	#for x in range(0,shape[0]):
	#	for y in range(0,shape[1]):
	#		sum = sum + diff[x][y]r + g + b
	diff = 0
	for channel in range(0,3):
		diff += sum(sum(abs( frame[:,:,channel] - prev_frame[:,:,channel])))

	return diff

	
	
def extract_MFCC(mfcc_data, start_samp, samp_per_window, total_samples):
	nr_of_MFCCs	= int(mfcc_data[0]) / 13

	# how far along are we?
	start 		= start_samp / float(total_samples)
	end 		= (start_samp + samp_per_window) / float(total_samples)

	# how much is that in complete, 13 coefficient, mfccs
	mfcc_nr_start	= int(start*nr_of_MFCCs / 13) * 13
	mfcc_nr_end		= mfcc_nr_start + 13	

	# retrieve the data
	mfcc_data	= mfcc_data[1:] # skip the first number
	mfcc		= mfcc_data[mfcc_nr_start:mfcc_nr_end]

	return mfcc

def extract_audio_energy(frame):
	D = 0 
	for i in range(0, len(frame)):
		D += abs(frame[i] / 100) *  abs(frame[i] / 100)
	
	return D

"""
This function processes a file by extracting audio features. 
"""
def process_audio(file, f_features):
	rate, ampl = read(file)
	data = []
	
	# we extract the mfcc data from precomputed files
	mfcc_file 	= file[:len(file)-3]+"mfcc"
	f 			= open(mfcc_file)	
	mfcc_data	= f.readline().split()
	samples_per_window 	= 1024
	start_sample 		= 0
	end_sample			= start_sample + samples_per_window
	# uncomment the line below if processing the whole file takes too long
	max_samples			= min(100000,len(ampl))
	#max_samples 		= len(ampl)
	
	while end_sample < max_samples:
		# You have to implement the functionality in extract_audio_energy
		if f_features == extract_audio_energy:
			feat = extract_audio_energy(ampl[start_sample:end_sample])
		
		# extract_MFCC already works
		if f_features == extract_MFCC:
			feat = extract_MFCC(mfcc_data, start_sample, samples_per_window, max_samples)
		
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
	
	frame_old		= None
	features_old	= []
	features_new	= []
	# Start extracting frames from video.
	while(index < frame_count):
		# read 1 frame of video
		# frame is a numpy.ndarray object
		features_old 	= features_new
		(success,frame) = capture.read()
		
		# Your code here...
		# steps:
		# 1) consider when we should exit the while loop
		# 2) extract the features of the current frame with f_features(frame)
		# 3) store the distance between the features of consecutive frames
				
		# Compute the distance between features of consecutive frames.
		# replace the [] with your distance calculation. Consider what information
		# you need (if any) from the previous frame to compute the feature
		# Hint: look at the respective feature function parameters.
		feat = []
		if f_features == extract_histogram and frame_old != None: # and len(features_old) > 0:
			feat = extract_histogram(frame, frame_old)
			
		if f_features == extract_temporal_difference and frame_old != None:
			feat = extract_temporal_difference(frame, frame_old)
		
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
This function implements a sliding window that can be used to compare a clip(window) and a video(signal) for each 
position of the window it will take the 2-norm of the differences between the window and the signal. The signal returned indicates the quallity of the match.
"""
def sliding_window(window,signal):
	wl	= len(window)
	sl	= len(signal)
	if sl-wl<0:
		print("Error: Query clip is longer than candidate signal")
		return ([],float("inf"),-1)

	diff	= numpy.zeros(sl-wl)
	minimum	= 9999
	frame	= -1
	# overlay the window starting with every frame in the video
	for i in range(0,sl-wl):
		# find the 2-norm of the differences between the signals and
		# store the minimum in the signal and at which frame it occurs
		diff[i] = []
	return (diff, minimum, frame)

"""
This function compares a given clip to all the videos in a database. It then plots the results and prints the best matching video and the best matching frame position.
"""
def compare(clip,database):
	print("[*] Comparing clip to database")
	match 	= 'Error'
	minimum	= 9999
	frame	= -1
	for video_name in database:
		print("  - " + video_name)
		candidate_video = database[video_name]
		# run sliding window over the video
		(diff, vid_minimum, vid_frame) = sliding_window(clip, candidate_video)

		# add difference signal to plot using plot.plot
		# add legend to plot
		# if a minimum is found, store it and its corresponding video and the location (frame)
	
	print("Found video " +match)
	print("Frame: " + str(frame))
	plot.show()
	# Print the name of the identified video and plot the difference signals for all videos
	# using plot.show()
	








"""
This function builds a database of feature signals that can then be processed later if needed. It does this by
processing all the media .ogv file in the folder /media/.
"""
def build_database():
	print("[*] Constructing database")
	database = {}
	for path in glob.glob("../media/*.ogv"):
		filename = ( regex.findall("/([^/]+)\.ogv",path) )[0]
		print(" [-] Parsing path: " + path)
		feature_functions = [extract_audio_features, extract_temporal_difference, extract_histogram]
		for feat_func in feature_functions:
			database[filename, feat_func.__name__] 	= process_video(path, feat_func)
		
		
	print("[*] Saving database to disk")	
	database_file = open('database.db','wb')
	pickle.dump(database,database_file)
	database_file.close()

"""
This function loads all the feature signals stored in a database file.
"""
def load_database():
	print("[*] Loading database from disk")
	database_file = open('database.db', 'rb')
	database = pickle.load(database_file)
	database_file.close()
	print("[*] Done")
	return database
