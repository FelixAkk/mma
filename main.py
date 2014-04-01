import numpy 
import matplotlib.pyplot as plot
import medialab as medialab

from pydub import AudioSegment
import wave
import time

def plot_feature_curve(filename, f_feat):
	#data = medialab.process_audio("../media/"+filename, f_feat)
	#plot.plot(range(0,len(data)), data)
	
	data = medialab.process_video("./../media/"+filename, f_feat)
	
	print(data)
	plot.plot(range(1,len(data)), data[1:])
	plot.show()
	
"""
This example extracts a clip of N frames from a video and compares it to all the 10 videos in the database. 
It then plots the resulting comparison signals and prints the video and frame were the clip fits best.
"""
def assignment02(query_video, modality_function_name, frameStart=400, clip_length=1000):
	database 	= medialab.load_database()
	frameEnd	= frameStart + 1000	
	
	vid = database[query_video, modality_function_name]
	if frameEnd > len(vid):
		print("Error: Unable to extract clip. End point is out of range")
		return
	
	clip = vid[frameStart:frameEnd]
	# plot the signal
	plot.plot(range(0,len(vid)), vid,'-k',label=query_video)
	plot.plot(range(frameStart,frameStart+len(clip)),clip,'-y',label="clip")
		
	# make a comparison of the clip to the database
	db = {}
	for video, modality in database:
		if modality == modality_function_name:
			db[video] = database[video, modality]
	medialab.compare(clip,db)

	
print("Multimedia Analysis - Video Lab")

#plot_feature_curve("./../media/video_03.ogv", medialab.extract_temporal_difference)
plot_feature_curve("video_07.ogv", medialab.extract_histogram)
#plot_feature_curve("video_02.wav", medialab.extract_audio_energy)


#medialab.build_database();
