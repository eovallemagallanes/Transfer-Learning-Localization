from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt


def loadimgs(filenames, target_size):
	images = np.zeros((len(filenames), target_size[0], target_size[1], target_size[2]))
	for i, filename in tqdm(enumerate(filenames)):
		img = image.load_img(filename, target_size=(target_size[0], target_size[1]))
		img_array = image.img_to_array(img)
		img_array = np.expand_dims(img_array, axis=0)
		images[i, :, :, :] = img_array.astype("float32") / 255.0
	return images


def loadvisualmap(path_dataset, target_size):
	# get path string
	imgs_path = [f for f in listdir(path_dataset) if isfile(join(path_dataset, f))]
	imgs_path.sort()

	# load and resize images
	train_filenames = [path_dataset + '/' + img_path for img_path in imgs_path]
	visual_map = loadimgs(train_filenames, target_size)

	# set labels
	classes = len(train_filenames)
	y_train = np.arange(classes)
	y_train = to_categorical(y_train, num_classes=classes, dtype='int32')

	return visual_map, y_train


def mapaugmentation(path_dataset, path_augmented, total_images=10):
	# define image data generator
	aug = ImageDataGenerator(
			rotation_range=10,
			zoom_range=0.15,
			width_shift_range=0.15,
			height_shift_range=0.15,
			shear_range=0.15,
			horizontal_flip=False,
			fill_mode="nearest")

	# get path string
	imgs_path = [f for f in listdir(path_dataset) if isfile(join(path_dataset, f))]
	imgs_path.sort()
	train_filenames = [path_dataset + '/' + img_path for img_path in imgs_path]

	for i, filename in tqdm(enumerate(train_filenames)):
		path = path_augmented + '/class_%03d' %i
		try:
			os.mkdir(path)
		except OSError:
			print ("Creation of the directory %s failed" % path)
		else:
			print ("Successfully created the directory %s " % path)
            
		#print(filename)
		img = image.load_img(filename)
		img_array = image.img_to_array(img)
		img_array = np.expand_dims(img_array, axis=0)

		imageGen = aug.flow(img_array, batch_size=1, save_to_dir=path,save_prefix="image_%03d" %i, save_format="jpg")

		# Define number of augmented image which you want to download and iterate through loop
		j = 1
		for e in imageGen:
			if (j == total_images):
				break
			j = j +1


def loadtestimages(path_dataset, target_size, path_ground_truth_idxs, path_train_idxs):
	path_imgs = [f for f in listdir(path_dataset) if isfile(join(path_dataset, f))]
	path_imgs.sort()

	test_filenames = [path_dataset+ '/' + img_path for img_path in path_imgs]
	test_imgs = loadimgs(test_filenames, target_size)

	gt_test_idx = np.genfromtxt(path_ground_truth_idxs,delimiter=',')
	train_idx = np.genfromtxt(path_train_idxs,delimiter=',')

	classes = len(train_idx)
	y_test = [(np.where(train_idx == gt)[0][0]) for gt in gt_test_idx]
	y_test = to_categorical(y_test, num_classes=classes, dtype='int32')

	return test_imgs, y_test, gt_test_idx, train_idx


def localization(model, test_img, train_idx, top_k = 3):
	pred = model.predict(test_img)
	#print(pred)
	top_k_idx = np.argsort(pred[0])[-top_k:]
	#print('Top k idx {}'.format(top_k_idx))
	top_k_values = [pred[0][i] for i in top_k_idx]
	r = []
	r.extend(top_k_idx)
	r.extend(top_k_values)
	r.extend(np.asarray(train_idx[top_k_idx], dtype=np.int32))

	return r


def plottopk(visual_map, query_img, loc_res, gt_idx, top_k = 3):
	fig = plt.figure(figsize=(10, 10))  # width, height in inches
	plot_imgs = [query_img]
	plot_imgs.extend(visual_map[loc_res[:top_k]])
	probs_txt = ['p = '+ str(prob) for prob in loc_res[top_k: -top_k]]
	all_x_labels = ['Query']
	all_x_labels.extend(probs_txt)
	all_y_labels = ['Ground-Truth index = %d' %gt_idx]
	all_y_labels.extend(['Index = ' + str(idx) for idx in loc_res[-top_k:]])

	for i in range(top_k+1):
		sub = fig.add_subplot(1, top_k+1, i + 1)
		sub.set(xlabel=all_x_labels[i], ylabel=all_y_labels[i])
		sub.imshow(plot_imgs[i])
		sub.set_xticks([])
		sub.set_yticks([])