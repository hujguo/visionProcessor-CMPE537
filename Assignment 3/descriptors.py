import cv2 as cv
import os

class HOG:
    def detectAndCompute(self, image, dummy=1):
        ## This function takes an RGB image as input. First, the image is turned into Gray scale. Then, HOG is calculated.
        resize=(128,128)
        grid_size=(8,8)
        num_of_bins=60

        grid_dim = int(resize[0]/grid_size[0])

        # Resize the image to a fixed size:
        image = cv.resize(image,resize)

        # Convert the BGR to grayscale format:
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

        # Create an empty array for the descriptor with proper dimensions:
        hist = np.zeros((grid_size[0]*grid_size[1] , num_of_bins))

        # Calculate the gradients in x and y axis:
        Gradx = cv.Sobel(image,cv.CV_32F,1,0,ksize=3) # By using cv2.CV_32F, gradient of each pixel will be 32-bit floating numbers
        Grady = cv.Sobel(image,cv.CV_32F,0,1,ksize=3)

        # Find the angles between the gradients in radians:
        GradRadian = np.arctan2(Grady,Gradx) # Each element is between -3.14 and 3.14

        # Create an array that contains centers of grids (dummy keypoints):
        dummy_kps = np.zeros((grid_size[0]*grid_size[1],2))

        # Scan the image by the windows and find the histogram of each window. Then, concatenate them column wise:
        for i in 