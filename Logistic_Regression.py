import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.ndimage as nd
from scipy import misc
from PIL import Image
from lr_utils import load_dataset
from model import model
from predict import predict


# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture 
index = 25
plt.imshow(train_set_x_orig[index])
#plt.show()
#print ("y = " + str(train_set_y[:,index]) + ", it's a '" + classes[np.squeeze(train_set_y[:,index])].decode("utf-8") +  "' picture.")

# Find the values for m_train (number of training examples), m_test (number of test examples)
# num_px (= height = width of a training image)
m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
#print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
#print ("train_set_y shape: " + str(train_set_y.shape))
#print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
#print ("test_set_y shape: " + str(test_set_y.shape))
#print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))

#Let's standardize our dataset.
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)




# (PUT YOUR IMAGE NAME) 
my_image = "animal.jpg"   # change this to the name of your image file 
## END CODE HERE ##

# We preprocess the image to fit your algorithm.
#fname = "images/" + my_image
fname = my_image
image = np.array(nd.imread(fname, flatten=False))
my_image = misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T

my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


