import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
import re
import random
from collections import Counter
import itertools 
from sklearn.decomposition import PCA


from os import listdir
"""
TODO 0: Find out the image shape as a tuple and include it in your report.
"""
IMG_SHAPE = (64,64)

def load_data(data_dir, top_n=10):
	"""
	Load the data and return a list of images and their labels.
	:param data_dir: The directory where the data is located
	:param top_n: The number of people with the most images to use
	Suggested return values, feel free to change as you see fit
	:return data_top_n: A list of images of only people with top n number of images
	:return target_top_n: Corresponding labels of the images
	:return target_names: A list of all labels(names)
	:return target_count: A dictionary of the number of images per person
	"""
	# read and randomize list of file names
	file_list = [fname for fname in listdir(data_dir) if fname.endswith('.pgm')]
	random.shuffle(file_list)
	name_list = [re.sub(r'_\d{4}.pgm', '', name).replace('_', ' ') for name in file_list]
	# get a list of all labels
	target_names = sorted(list(set(name_list)))
	# get labels for each image
	target = np.array([target_names.index(name) for name in name_list])
	# read in all images
	data = np.array([cv2.imread(data_dir + fname, 0) for fname in file_list])

	"""
	TODO 1: Only preserve images of 10 people with the highest occurence, then plot
	a histogram of the number of images per person in the preserved dataset.
	Include the histogram in your report.
	"""


	# YOUR CODE HERE
	# target_count is a dictionary of the number of images per person
	# where the key is an index to label ('target'), and the value is the number of images
	# Try to use sorted() to sort the dictionary by value, then only keep the first 10 items of the output list.
	target_count = {}
	target_count = Counter(target)
	# target_count = sorted(target_count)
	# print(type(target_count))
	# print(target_count)
	
	# data_top_n is a list of labels of only people with top n number of images
	top_n_ = target_count.most_common(top_n)
	
	name_ref = []
	# populating reference names for lookin at original files
	for i in range (len(top_n_)):
		name_ref.append((target_names[top_n_[i][0]],top_n_[i][0], (top_n_[i][1]) ))
	print(name_ref)

	
	print(type(top_n_))
	print(top_n_)
	target_top_n = []
	data_top_n = []
	for i in range(len(name_ref)):
		for j in range(len(file_list)):
			if(re.sub(r'_\d{4}.pgm', '', file_list[j]).replace('_', ' ') == name_ref[i][0]):
				target_top_n.append(i)
				data_top_n.append(data[j])

	name_labels_plt = []
	count_plt = []
	for i in range(len(name_ref)):
		name_labels_plt.append(name_ref[i][0])
		count_plt.append(name_ref[i][2])
	
	fig = plt.figure()
	plt.bar(name_labels_plt, count_plt)
	fig.autofmt_xdate()
	plt.show() #uncomment to show plot

	return data_top_n, target_top_n, target_names, target_count


def load_data_nonface(data_dir):
	"""
	Your can write your functin comments here.
	"""
	"""
	TODO 2: Load the nonface data and return a list of images.
	"""
	# YOUR CODE HERE
	# Take a look at the load_data() function for reference

	file_list = [fname for fname in listdir(data_dir) if fname.endswith('.png')]
	data = np.array([cv2.imread(data_dir + fname, 0) for fname in file_list])
	# print("non-face data size" )
	# print(len(data))
	return data

def perform_pca(data_train, data_test, data_noneface, n_components, plot_PCA=False):
	"""
	Your can write your functin comments here.
	"""
	"""
	TODO 3: Perform PCA on the training data, then transform the training, testing,
	and nonface data. Return the transformed data. This includes:
	a) Flatten the images if you haven't done so already
	b) Standardize the data (0 mean, unit variance)
	c) Perform PCA on the standardized training data
	d) Transform the standardized training, testing, and nonface data
	e) Plot the transformed training and nonface data using the first three
	principal components if plot_PCA is True. Include the plots in your report.
	f) Return the principal components and transformed training, testing, and nonface data
	"""
	# YOUR CODE HERE
	# flattening images
	data_train_flat = []
	data_test_flat = []
	data_nonface_flat = []
	print("-----PCA FUNCTION-----")
	for img in data_train:
		data_train_flat.append(img.flatten())
	for img in data_test:
		data_test_flat.append(img.flatten())
	for img in data_noneface:
		data_nonface_flat.append(img.flatten())
	# print(len(data_train_flat))
	# print((data_train_flat[0].shape))
	# print((data_nonface_flat[0].shape))

	# You can use the StandardScaler() function to standardize the data
	scaler = StandardScaler()
	data_train_centered = scaler.fit_transform(data_train_flat)
	data_test_centered = scaler.transform(data_test_flat)
	data_noneface_centered = scaler.transform(data_nonface_flat)
	# print(len(data_train_centered))
	# print(type(data_train_centered[0]))
	# print((data_train_centered[0].shape))
	# print((data_train_centered[0][0]))
	# print('-----')

	# You can use the decomposition.PCA() and function to perform PCA
	# You can check the example code in the documentation using the links below
	# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
	pca = PCA(n_components = n_components)
	# You can use the pca.transform() function to transform the data
	data_train_pca = pca.fit_transform(data_train_centered)
	data_test_pca = pca.transform(data_test_centered)
	data_noneface_pca = pca.transform(data_noneface_centered)
	# print(len(data_train_pca))
	# print(type(data_train_pca[0]))
	# print((data_train_pca[0].shape))
	# print((data_train_pca[0][0]))

	# print('-0---')
	# print(data_train_pca[:,0])

	# You can use the scatter3D() function to plot the transformed data
	# Please not that 3 principal components may not be enough to separate the data
	# So your plot of face and nonface data may not be clearly separated
	if plot_PCA:
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter3D(data_train_pca[:,0], data_train_pca[:,1], data_train_pca[:,2],  label='Training Data', c='b' )
		ax.scatter3D(data_noneface_pca[:,0], data_noneface_pca[:,1], data_noneface_pca[:,2],  label='Nonface Data', c='g' )
		ax.set_xlabel('PC 1')
		ax.set_ylabel('PC 2')
		ax.set_zlabel('PC 3')
		ax.legend()
		plt.title('PCA of Training and Nonface Data')
		plt.show()

	return pca, data_train_pca, data_test_pca, data_noneface_pca


def plot_eigenfaces(pca):
	"""
	TODO 4: Plot the first 8 eigenfaces. Include the plot in your report.
	"""
	print(pca.components_.shape)
	
	n_row = 2
	n_col = 4
	fig, axes = plt.subplots(n_row, n_col, figsize=(12, 6))
	for i in range(n_row * n_col):
		
		# YOUR CODE HERE
		# The eigenfaces are the principal components of the training data
		# Since we have flattened the images, you can use reshape() to reshape to the original image shape
		plt.subplot(n_row, n_col, i+1)
		plt.imshow(pca.components_[i].reshape(IMG_SHAPE), cmap='gray')
		
	plt.show()


def train_classifier(data_train_pca, target_train):
	"""
	TODO 5: OPTIONAL: Train a classifier on the training data.
	SVM is recommended, but feel free to use any classifier you want.
	Also try using the RandomizedSearchCV to find the best hyperparameters.
	Include the classifier you used as well as the parameters in your report.
	Feel free to look up sklearn documentation and examples on usage of classifiers
	"""
	# YOUR CODE HERE
	# You can read the documents from sklearn to learn about the classifiers provided by sklearn
	# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
	# If you are using SVM, you can also check the example below
	# https://scikit-learn.org/stable/modules/svm.html
	# Also, you can use the RandomizedSearchCV to find the best hyperparameters
	clf = None
	return clf
if __name__ == '__main__':
	"""
	Load the data
	Face Dataset from https://conradsanderson.id.au/lfwcrop/
	Modified from original dataset http://vis-www.cs.umass.edu/lfw/
	Noneface Dataset modified from http://image-net.org/download-images
	All modified datasets are available in the Box folder
	"""
	data, target, target_names, target_count = load_data('ps8a-dataset/lfw_crop/', top_n=10)
	data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.25, random_state=42)
	data_noneface = load_data_nonface('ps8a-dataset/imagenet_val1000_downsampled/')
	print("Total dataset size:", len(data))
	print("Training dataset size:", len(data_train))
	print("Test dataset size:", len(data_test))
	print("Nonface dataset size:", len(data_noneface))

	# Perform PCA, you can change the number of components as you wish
	pca, data_train_pca, data_test_pca, data_noneface_pca = perform_pca(
		data_train, data_test, data_noneface, n_components=8, plot_PCA=True
		)
	


	# Plot the first 8 eigenfaces. To do this, make sure n_components is at least 8
	plot_eigenfaces(pca)
	
	
	
	
	"""
	Start of PS 8-2
	This part is optional. You will get extra credits if you complete this part.
	"""
	# Train a classifier on the transformed training data
	classifier = train_classifier(data_train_pca, target_train)
	# Evaluate the classifier
	pred = classifier.predict(data_test_pca)
	# Use a simple percentage of correct predictions as the metric
	accuracy = np.count_nonzero(np.where(pred == target_test)) / pred.shape[0]
	print("Accuracy:", accuracy)
	"""
	TODO 6: OPTIONAL: Plot the confusion matrix of the classifier.
	Include the plot and accuracy in your report.
	You can use the sklearn.metrics.ConfusionMatrixDisplay function.
	"""
	# YOUR CODE HERE
	"""
	TODO 7: OPTIONAL: Plot the accuracy with different number of principal components.
	This might take a while to run. Feel free to decrease training iterations if
	you want to speed up the process. We won't set a hard threshold on the accuracy.
	Include the plot in your report.
	"""
	n_components_list = [3, 5, 10, 20, 40, 60, 80, 100, 120, 130]
	# YOUR CODE HERE