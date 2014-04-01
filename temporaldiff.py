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
