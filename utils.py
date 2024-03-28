from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import correlate
import sklearn.cluster
from scipy.spatial.distance import cdist

def computeHistogram(img_file, F, textons):
    # Read the image
    image = io.imread(img_file)
    
    # Convert the image to grayscale
    img_gray = rgb2gray(image)
    
    # Apply filter bank to the grayscale image
    filter_responses = np.zeros((img_gray.shape[0], img_gray.shape[1], F.shape[2]))
    for i in range(F.shape[2]):
        filter_responses[:, :, i] = correlate(img_gray, F[:, :, i])
    
    # Reshape filter_responses to have 2D shape
    filter_responses_2d = filter_responses.reshape(-1, filter_responses.shape[2])
    
    # Transpose filter_responses_2d to match the number of features in textons
    filter_responses_2d = filter_responses_2d.T

    # some issue here in the matrix size.
    # print("Shape of filter_responses_2d:", filter_responses_2d.shape)
    # print("Shape of textons:", textons.shape)
    
    # Compute distances between filter responses and textons
    distances = cdist(filter_responses_2d, textons)
    
    # Assign each pixel to the nearest texton
    assignments = np.argmin(distances, axis=1)
    
    # Compute histogram representation
    histogram, _ = np.histogram(assignments, bins=len(textons), range=(0, len(textons)))
    
    return histogram

    
def createTextons(F, file_list, K):
      ### YOUR CODE HERE

    filter_response = []
    for img_file in file_list:
        img = io.imread(img_file)
        img_gray = rgb2gray(img)
        
        for i in range(F.shape[2]):
            filter_response.append(correlate(img_gray, F[:, :, i]).flatten())

    
    kmeans = sklearn.cluster.KMeans(n_clusters=K)
    kmeans.fit(filter_response)

    textons = kmeans.cluster_centers_

    return textons
    ### END YOUR CODE
