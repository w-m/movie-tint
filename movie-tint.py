import cv2
import numpy as np
import math, sys
from progressbar import Bar, ETA, Percentage, ProgressBar, Widget
from time import time
from itertools import izip

# Call with an input movie file and output file name,
# e.g. python movie-tint my-movie.avi my-tint.png

# how many movie frames should one (horizontal) pixel cover?
FRAMES_PER_PIXEL = 48

# display progress window?
# updates whole image, not very efficient, disable for very large images
SHOW_PROGRESS = True

# select a clip
START_FRAME = 0
END_FRAME = sys.maxint

# downscaled frame height in pixels
# increase this for more precision, esp. when FRAMES_PER_PIXEL is low
# decrese to speed up calculation
FRAME_HEIGHT = 16

def kmeans(image):
	Z = np.float32(image)
	K = 3
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
	ret, labels, cluster_centers = cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
	bins = np.bincount(labels.flatten()).astype('float')
	s = sum(bins)
	bins /= s
	return zip(bins, cluster_centers)

def hue_shift(input_image, degrees):
	""" Shift hue of the image by a (positive) amount of degrees
		--> Input between 0 and 179 (OpenCV HSV range)"""
	image = input_image if 180 + degrees <= 255 else input_image.astype('uint16')
	image[:, :, 0] += degrees
	image[:, :, 0] %= 180
	return image

class FPS(Widget):
	""" Show frames per second in progress bar"""
	def __init__(self):
		self.last_update = time()
		self.last_frames = self.total_frames = 0
		self.last_fps = -1

	def update(self, pbar):
		current_time = time()
		fps = (self.total_frames - self.last_frames) / (current_time - self.last_update)
		if (self.last_fps > 0):
			fps = 0.05 * fps + 0.95 * self.last_fps
		self.last_fps = fps
		self.last_frames = self.total_frames
		self.last_update = current_time
		return "Frames per second: %d" % fps

class FPSProgressBar():
	def __init__(self, number_of_chunks, chunk_size):
		self.fps_widget = FPS()
		self.chunk_size = chunk_size
		widgets = ['Progress: ', Percentage(), ' ', Bar(), ' ', self.fps_widget, ' | ', ETA()]
		self.pbar = ProgressBar(widgets=widgets, maxval=number_of_chunks).start()

	def update(self, chunk_count):
		self.fps_widget.total_frames += self.chunk_size
		self.pbar.update(chunk_count)

	def finish(self):
		self.pbar.finish()

def draw_gauss(radius, color):
	"""Draw a blurry circle in an hsv color"""
	hue, sat, val = color
	circle_env = radius * 2
	circle = np.ndarray((circle_env * 2 + 1, circle_env * 2 + 1, 4), dtype = np.uint8)

	# alpha will be blurred outside the circle
	# hue, sat need to be constant everywhere
	circle[:,:,:] = [hue, sat, val, 0]
	cv2.circle(circle, (circle_env, circle_env), radius, [hue, sat, val, 255], -1)

	gauss_size = max(radius, 3)
	if gauss_size % 2 == 0:
		gauss_size += 1

	# blur alpha
	circle[:, :, 3] = cv2.GaussianBlur(circle[:, :, 3], (gauss_size, gauss_size), 0)
	return circle

def draw_overlay(radius, color, sub_result):
	"""Draw blurry circles, blend them with the background image"""
	blurred_circle = draw_gauss(radius, color).reshape((-1, 4))
	flat_sub_result = sub_result.reshape((-1, 4))

	result_alpha = flat_sub_result[:,3].astype('float')
	circle_alpha = blurred_circle[:,3].astype('float')
	alpha_sum = result_alpha + circle_alpha
	alpha = circle_alpha / alpha_sum
	alpha_inv = 1 - alpha

	new_alpha = np.minimum(alpha_sum, 255)

	hue_channel = blurred_circle[:,0] * alpha + flat_sub_result[:,0] * alpha_inv
	sat_channel = blurred_circle[:,1] * alpha + flat_sub_result[:,1] * alpha_inv
	val_channel = blurred_circle[:,2] * alpha + flat_sub_result[:,2] * alpha_inv

	chans = np.dstack((hue_channel.astype('uint8'), sat_channel.astype('uint8'), val_channel.astype('uint8'), new_alpha.astype('uint8')))
	return chans.reshape(sub_result.shape)

class Capture():
	"""Read frames from an input file, resize them"""
	def __init__(self, file_name, frame_height):
		self.cap = cv2.VideoCapture(file_name)	
		self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, START_FRAME)
		h = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
		w = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
		self.aspect_ratio = float(h) / w
		self.new_h = frame_height
		self.new_w = int(self.new_h / self.aspect_ratio)

		self.calc_chunk_size()

	def calc_chunk_size(self):
		self.frame_count = min(END_FRAME, self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) - START_FRAME
		self.chunk_size = FRAMES_PER_PIXEL
		self.number_of_chunks = int(self.frame_count / self.chunk_size)
		self.height = self.number_of_chunks * self.aspect_ratio
		print "frames: %d, chunks: %d, frames per chunk: %d" % (self.frame_count, self.number_of_chunks, self.chunk_size)
		print "result image: %d x %d px" % (self.number_of_chunks, self.height)

	def read_frame(self):
		f, image = self.cap.read()
		if not f: return False, None
		smaller = cv2.resize(image, (self.new_w, self.new_h))
		return True, smaller #.reshape((-1,3))

	def read_chunk(self):
		first_frame = True
		for current_frame in range(self.chunk_size):
			success, frame = self.read_frame()
			if not success:
				if not first_frame:
					break
				return False, None

			if first_frame:
				frame_data = frame
				first_frame = False
			else:
				frame_data = np.concatenate((frame_data, frame), axis = 0)
		return True, frame_data

	def chunks(self):
		for chunk_count in range(self.number_of_chunks):
			success, frame_data = self.read_chunk()
			if not success:
				break
			yield frame_data

class MovieHistogram():

	def __init__(self, file_name, output_file):
		
		self.output_file = output_file

		self.capture = Capture(file_name, FRAME_HEIGHT)

		self.px_per_hue_class = self.capture.height / 180
		self.circle_radius = self.px_per_hue_class * 2
		self.offset = self.circle_radius * 4

		self.pbar = FPSProgressBar(self.capture.number_of_chunks, self.capture.chunk_size)
		self.process_file()

	def create_output(self):

		result = self.result_image[:,:,:3]
		back_shift = hue_shift(result, 150).astype('uint8')
		converted = cv2.cvtColor(back_shift, cv2.COLOR_HSV2BGR)

		alpha = self.result_image[:,:,3].astype('float') / 255
		alpha_inv = 1 - alpha

		r, b, g = 0, 0, 0

		converted[:,:,0] = converted[:,:,0] * alpha + b * alpha_inv
		converted[:,:,1] = converted[:,:,1] * alpha + g * alpha_inv
		converted[:,:,2] = converted[:,:,2] * alpha + r * alpha_inv

		return converted

	def finish(self):
		cv2.destroyAllWindows()
		self.pbar.finish()
		return cv2.imwrite(self.output_file, self.create_output())
	
	def process_chunk(self, chunk_count, frame_data):

		for weight, hsv in kmeans(frame_data):
			hue, sat, val = hsv

			relative_hue_position = (hue * self.px_per_hue_class +
									self.px_per_hue_class / 2)
			draw_hue = relative_hue_position + self.offset
			color = map(int, hsv)

			radius = int(math.ceil(self.circle_radius * weight))

			if radius > 0:
				circle_env = radius * 2
				center = (draw_hue, chunk_count + self.offset)
				sub_result = self.result_image[center[0] - circle_env : center[0] + circle_env + 1,
				 			              center[1] - circle_env : center[1] + circle_env + 1, :]
				flat_sub_result = draw_overlay(radius, color, sub_result)

				self.result_image[center[0] - circle_env : center[0] + circle_env + 1,
			                center[1] - circle_env : center[1] + circle_env + 1, :] = flat_sub_result

	def process_file(self):

		self.result_image = np.zeros((self.capture.height + self.offset * 2,
									  self.capture.number_of_chunks + self.offset * 2, 4), np.uint8)

		# will be doing div by zero in draw_gauss
		np.seterr(invalid='ignore')

		for chunk_count, frame_data in enumerate(self.capture.chunks()):
			hsv = cv2.cvtColor(frame_data, cv2.COLOR_BGR2HSV)
			# + 60 degrees --> line up red
			shifted = hue_shift(hsv, 30)  
			flat = shifted.reshape((-1,3))
			self.process_chunk(chunk_count, flat)

			self.pbar.update(chunk_count)

			if SHOW_PROGRESS:
				cv2.imshow("progress", self.create_output())
				cv2.waitKey(1)

		self.finish()

MovieHistogram(sys.argv[1], sys.argv[2])
