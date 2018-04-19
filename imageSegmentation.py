#Image Quantization

import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

K = 4

image = cv2.imread('inputFileImages//image1.jpg')
Z = image.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteriaOfimages, number of clusters(K) and apply kmeans()
criteriaOfimages = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret,criteriaLabel,center=cv2.kmeans(Z,K,None,criteriaOfimages,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[criteriaLabel.flatten()]
output_image = res.reshape((image.shape))
cv2.imwrite('outputClusteredImages//output1_'+str(K)+'.png', output_image);

print("Image1 has been clustered")


image = cv2.imread('inputFileImages//image2.jpg')
Z = image.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteriaOfimages, number of clusters(K) and apply kmeans()
criteriaOfimages = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret,criteriaLabel,center=cv2.kmeans(Z,K,None,criteriaOfimages,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[criteriaLabel.flatten()]
output_image = res.reshape((image.shape))
cv2.imwrite('outputClusteredImages//output2_'+str(K)+'.png', output_image);
print("Image2 has been clustered")





image = cv2.imread('inputFileImages//image3.jpg')
Z = image.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteriaOfimages, number of clusters(K) and apply kmeans()
criteriaOfimages = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret,criteriaLabel,center=cv2.kmeans(Z,K,None,criteriaOfimages,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[criteriaLabel.flatten()]
output_image = res.reshape((image.shape))
cv2.imwrite('outputClusteredImages//output3_'+str(K)+'.png', output_image);
print("Image3 has been clustered")



image = cv2.imread('inputFileImages//image4.jpg')
Z = image.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteriaOfimages, number of clusters(K) and apply kmeans()
criteriaOfimages = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret,criteriaLabel,center=cv2.kmeans(Z,K,None,criteriaOfimages,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[criteriaLabel.flatten()]
output_image = res.reshape((image.shape))
cv2.imwrite('outputClusteredImages//output4_'+str(K)+'.png', output_image);
print("Image4 has been clustered")




image = cv2.imread('inputFileImages//image5.jpg')
Z = image.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteriaOfimages, number of clusters(K) and apply kmeans()
criteriaOfimages = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

ret,criteriaLabel,center=cv2.kmeans(Z,K,None,criteriaOfimages,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[criteriaLabel.flatten()]
output_image = res.reshape((image.shape))
cv2.imwrite('outputClusteredImages//output5_'+str(K)+'.png', output_image);
print("Image5 has been clustered")