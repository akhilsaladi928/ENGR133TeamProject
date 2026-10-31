"""
Course Number: ENGR 133
Semester: Fall 2026

Description:
    To create a program which measures the tempature and
    pressure of an extraction vessel to provide warnings
    and allow the values to stay in the acceptable range

Assignment Information:
    Assignment:     tp2 team 1
    Team ID:        LC4 - 18
    Author:         Thomas Morgan, morga487@purdue.edu
    Date:           9/17/2026

Contributors:
    Name, login@purdue [repeat for each]

    My contributor(s) helped me:
    [ ] understand the assignment expectations without
        telling me how they will approach it.
    [ ] understand different ways to think about a solution
        without helping me plan my solution.
    [ ] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""

import numpy as np
import math as m
import matplotlib.pyplot as plt
from src.tp1_team_3_18 import create_gaussian_kernel, gaussian_filter, rgb_to_grayscale
from tp1_team_1_18 import load_img

#Sobel matrix for the sobel calultions in the X directions
SOBELX = [[-1,0,1],
          [-2,0,2],
          [-1,0,1]]

#Sobel matrix for the sobel calculations in the Y directions
SOBELY = [[-1,-2,-1],
          [0,0,0],
          [1,2,1]]

#threshold local global in case we need to change to adjust
THRESHOLD = 50


def sobel_filter(data):
    #convert to np array of floats bc sobel filter works with manipulating floats
    data_array = np.array(data, dtype=float)
    #Zero pad
    data_array = np.pad(data_array, 1, mode='constant')
    output = []
    rows, cols = data_array.shape

    for r in range(1, rows-1):
        row = []
        for c in range(1, cols-1):
            #get the window from original image
            window = data_array[r-1:r+2, c-1:c+2]
            #get the sobel in the x direction
            arrayX = window * SOBELX
            #get the sobel in the y direction
            arrayY = window * SOBELY
            #Calculate magnitude of the gradient
            gx = np.sum(arrayX)
            gy = np.sum(arrayY)
            gradient_magnitude = m.sqrt(gx ** 2 + gy ** 2)

            #if magnitude is greater than 255 cap it at 255
            gradient_magnitude = min(gradient_magnitude, 255)

            #set value either to 255 if above threshold (white) or 0 if below (black)
            if gradient_magnitude < THRESHOLD:
                gradient_magnitude = 0
            else:
                gradient_magnitude = 255

            row.append(gradient_magnitude)
        output.append(row)

    #return image as numpy array
    return np.array(output).astype(np.uint8)



def main():
    #Get image from path
    path = input("Enter the path to the image file: ")

    img_array = (load_img(path)*255).astype(np.uint8)
    #convert to grayscale
    test_image = rgb_to_grayscale(img_array)
    #apply gaussian filter
    test_image = gaussian_filter(test_image, 1)
    #apply sobel filter
    test_image = sobel_filter(test_image)
    #present the image to see if we need to change threshold
    plt.imshow(test_image, cmap='gray', vmin=0, vmax=255)
    plt.show()
    return (0)

if __name__ == "__main__":
    main()
