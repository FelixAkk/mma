"""
This function extracts features from a given frame. It extracts a normalized histogram for each of the color channels
of the frame. 
"""
def extract(frame, prev_frame):
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
	
