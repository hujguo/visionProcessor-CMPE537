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

        # Calculate the gradients in x an