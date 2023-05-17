import pandas as pd
import cv2 as cv2
import numpy as np
import json
import cProfile
import pstats
import multiprocessing as mp
from io import BytesIO
from csv import writer 
import csv
import subprocess
from dataset.preprocessing import Preprocessing


class FeatureExtraction:
	def __init__(self, coco_file):
		self.coco_file = coco_file
		self.coco_data = None
		self.videos = []
		self.categories = []
		self.annotations = []
		self.video_columns = []
		self.category_columns = []
		self.features = ["Frame", "BoudingBox","Area","Circularity","Convexity","Rectangularity", "Elongation","Eccentricity","Solidity"]
		self.columns = self.video_columns+self.category_columns+self.features
		self.dataframe = None
		self.num_processes = mp.cpu_count()
		self.result_queue = mp.Queue()
		self.result_list = mp.Manager().list()
		self.lock = mp.Lock()
		self.semaphore = mp.Semaphore(self.num_processes)

	def run(self):
		
		# Start profiling
		pr = cProfile.Profile()
		pr.enable()

		# Load the COCO-formatted file
		with open(self.coco_file, 'r') as f:
			self.coco_data = json.load(f)
		
		# Extracting the different dictionaries within the coco file
		self.videos = self.coco_data['videos']
		self.categories = self.coco_data['categories']
		self.annotations = self.coco_data['annotations']

		# Create dataframe with the correct columns
		self.frameGenerator()

		# Extracting features based on the coco file and adding them to the dataframe
		self.extractFeatures()


		print(self.dataframe)

		# Stop profiling
		pr.disable()

		# Print profiling stats
		ps = pstats.Stats(pr)
		ps.sort_stats(pstats.SortKey.TIME)
		ps.print_stats(10)

	def frameGenerator(self):
		# Add columns for video and categories
		for data in self.coco_data:
			if data != 'annotations':
				for element in self.coco_data[data]:
					if element['id'] == 0:
						for key, value in element.items():
							if key == "id":
								if data == 'videos':
									self.video_columns.append('video_id')
								else:
									self.category_columns.append('category_id')
							elif key == "name":
								if data == 'videos':
									self.video_columns.append(key)
								else:
									self.category_columns.append('category_name')
							else:
								if data == 'videos':
									self.video_columns.append(key)
								else:
									self.category_columns.append(key)

		self.columns = self.video_columns+self.category_columns+self.features
		self.dataframe = pd.DataFrame(columns=self.columns)

	def preprocessFrame(self, video_path, frame):
		preprocess = Preprocessing(video_path, frame)
		return preprocess.frame, preprocess.mask

	def boundingBox(self, contour):
		x, y, w, h = cv2.boundingRect(contour)
		return [x,y,w,h]

	def area(self, contour, width, height, pct=False):
		if pct:
			return cv2.contourArea(contour)/(width*height)
		else:
			return cv2.contourArea(contour)
	def circularity(self, contour, area):
		"""
		Calculate the circularity of the contour
		"""
		perimeter = cv2.arcLength(contour, True)
		if perimeter != 0:
			return (2 * np.sqrt(np.pi * area)) / perimeter
		else:
			return None

	def convexity(self, contour, mask):
		"""
		Calculate the convexity of the contour
		"""
		try:
			perimeter = cv2.arcLength(contour, True)
			hull = cv2.convexHull(contour)
			hull_perimeter = cv2.arcLength(hull, True)
			return hull_perimeter / perimeter
		except cv2.error:
			return None

	def rectangularity(self, contour, area):
		"""
        Calculate the rectangularity of the contour
        """
		try:
			rect = cv2.minAreaRect(contour)
			rect_area = rect[1][0] * rect[1][1]
			rectangularity = area / rect_area
			return rectangularity
		except cv2.error:
			return None

	def elongation(self, contour):
		"""Calculate the elongation of a contour"""
		try:
			x, y, w, h = cv2.boundingRect(contour)
			if w >= h:
			    return h / w
			else:
			    return w /h

		except cv2.error:
			print("Error: Failed to compute elongation for contour")
			return None

	def eccentricity(self, contour):
		"""Calculate the eccentricity of a contour"""

		try:
			(x, y), (a, b), angle = cv2.fitEllipse(contour)
			if a >= b:
				return np.sqrt(1 - (b/2 / a/2) ** 2)
				#return b / a
			else:
				return np.sqrt(1 - (a / b) ** 2)
				#return a / b

		except cv2.error:
			print("Error: Failed to compute eccentricity for contour")
			return None

	def solidity(self, area, mask):
		"""
		Calculate the solidity of the contour
		"""
		if area > 0:
			indices = np.transpose(np.nonzero(mask))
			if indices.size > 0:
				try:
					hull = cv2.convexHull(indices)
					if len(hull) > 0:
						hull_area = cv2.contourArea(hull)
						return area / hull_area
				except cv2.error:
					pass
					return None

	def videoWorker(self,video):
		self.semaphore.acquire()

		video_annotations = []
		# For every key in the video annotation
		for key, value in video.items():
			
			# Checking if it is the video ID in the video dictionary
			if key == 'video_id':
				
				# Finding the video dictionary that has the corresponding ID
				video_dict = next(item for item in self.videos if item["id"] == value)
				
				# Making a start for all annotations for video ID == value rows
				video_data = list(video_dict.values())

			# Checking if it is category annotations within the video dictionary
			if key == 'catagories':
				
				# For annotations for each category
				for i, category in enumerate(value):
					
					# Check if there is any annotations for this category
					if category:
						
						# Taking each section of annotations
						category_dict = next(item for item in self.categories if item["id"] == i)
						category_data = list(category_dict.values())
						name = category_dict['name']
						#print(f'Annotations for category {i}: {name}')
						# For each section of annotations
						for section in category:
							
							# Find the total number of frames in section
							sec_frames = section['frame_end'] - section['frame_start'] + 1
							
							# For each frame in the annotation do this
							for frame in range(section['frame_start'], (section['frame_end']+1)):
								# Create row prefilled with video and category data
								row = video_data+category_data
								
								# Append the frame we are working on
								row.append(frame)

								# Apply preprocessing to the frame:
								preprocessed_frame, mask = self.preprocessFrame(video_data[1],frame)
								
								# Find the contours of the binary image
								contours, hierarchy = cv2.findContours(preprocessed_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

								# Do a area calculation on all contours
								size_check = []
								for contour in contours:
									size_check.append(self.area(contour, row[2], row[3]))

								# While there is more than 1 contour, remove smallest area
								while len(contours)>1:

									min_index = size_check.index(min(size_check))
									size_check.pop(min_index)
									contours = tuple([x for i, x in enumerate(contours) if i != min_index])


								actual_area = None
								if (len(contours) == 1) and (np.average(preprocessed_frame) != 0):
									# Adding all feature functions to a list:
									args = [contours[0]]
									feature_functions = [
															{'f': self.boundingBox, 'a': args},
															{'f': self.area, 'a': args},
															{'f': self.circularity, 'a': args},	
															{'f': self.convexity, 'a': args},
															{'f': self.rectangularity, 'a': args},
															{'f': self.elongation, 'a': args},
															{'f': self.eccentricity, 'a': args},
															{'f': self.solidity, 'a': args},									
													 	]
									# Run functions to extract each feature and appending it to row
									for i, function in enumerate(feature_functions):
										if i ==0:
											boundingBox = function['f'](*function['a'])
											if (0 in boundingBox) or (boundingBox[2] == row[2]) or (boundingBox[3] == row[3]):
												for x in range(len(self.features)-1):
													row.append(0)
												# This is where i want a break
												break
											else:
										  		row.append(boundingBox)
										
										if i ==1:
											row.append(function['f'](*function['a'], row[2], row[3], pct=True))
											actual_area = function['f'](*function['a'], row[2], row[3])
										elif (i == 2) or (i == 4):
											row.append(function['f'](*function['a'], actual_area))
										elif (i == 3):
											row.append(function['f'](*function['a'], mask))
										elif i == 7:
											row.append(function['f'](actual_area, mask))
										else:
											row.append(function['f'](*function['a']))
								else:
									for x in range(len(self.features)-1):
										row.append(0)
								video_annotations.append(row)

					# If there is no annotations for this category 
					else:
						category_dict = next(item for item in self.categories if item["id"] == i)
						name = category_dict['name']
						#print(f'There is no annotations for category id {i}: {name}')
		for key, value in video.items():
			# Checking if it is the video ID in the video dictionary
			if key == 'video_id':
				with self.lock:
					self.result_list.append(video_annotations)
		self.semaphore.release()


	def extractFeatures(self):

		# create a shared memory array to store the results
		result_array = mp.RawArray('i', len(self.annotations))
		
		# create a process for each video
		processes = []

		for video in self.annotations:
			p = mp.Process(target=self.videoWorker, args=(video,))
			processes.append(p)
			p.start()

		# Wait for all processes to finish
		for p in processes:
		    p.join()

		output = open('./featuresExtracted_no_touchy.csv','w')
		csv_writer = writer(output)
		self.result_list = list(self.result_list)
		csv_writer.writerow(self.columns)
		for video in self.result_list:
			for row in video:
				csv_writer.writerow(row)
		
		subprocess.run(["bash", "-c", "sort -t',' -k1,1n -k8,8n -k13,13n featuresExtracted_no_touchy.csv > featuresExtractedSorted.csv"])
		subprocess.run(["mv", "featuresExtractedSorted.csv", "featuresExtracted_no_touchy.csv"])
		with open('featuresExtracted.csv', 'rt') as f:
			reader = csv.reader(f)
			self.dataframe = pd.DataFrame(reader)
		self.dataframe.columns = self.dataframe.iloc[0]
		self.dataframe = self.dataframe[1:]
		# Check for duplicate frames, i.e. where both a cage and a person is within the frame
		subset_cols = ['Frame','BoudingBox','Area','Circularity','Convexity','Rectangularity','Elongation','Eccentricity','Solidity']
		self.dataframe = self.dataframe.drop_duplicates(subset=subset_cols, keep='first')

		# Save the new csv file with no doubles and a ['Cage'] column
		self.dataframe.to_csv('featuresExtracted_no_touchy_noDoubleFrames.csv',index=False)
if __name__=="__main__":
	FeatureExtractor = FeatureExtraction('dataset/video.json')
	FeatureExtractor.run()