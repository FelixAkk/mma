def extract(frame, mfcc_data, start_sample, samples_per_window, max_samples):
	D = 0 
	for i in range(0, len(frame)):
		D += abs(frame[i] / 100) *  abs(frame[i] / 100)
	return D
