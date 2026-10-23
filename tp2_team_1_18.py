"""
Course Number: ENGR 133
Semester: Fall 2026

Description:
    To create a program which measures the tempature and
    pressure of an extraction vessel to provide warnings
    and allow the values to stay in the acceptable range

Assignment Information:
    Assignment:     Assingment
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
from tp1_team_3_18 import create_gaussian_kernel, gaussian_filter, rgb_to_grayscale
from tp1_team_1_18 import load_img

SOBELX = [[-1,0,1],
          [-2,0,2],
          [-1,0,1]]
SOBELY = [[-1,-2,-1],
          [0,0,0],
          [1,2,1]]

THRESHOLD = 50

def sobel_filter(data):
    data_array = np.array(data, dtype=float)
    output = []
    row, col = data_array.shape
    
    for r in range(1, row-1):
        row = []
        for c in range(1, col-1):
            window = data_array[r-1:r+2, c-1:c+2]
            arrayX = window * SOBELX
            arrayY = window * SOBELY
            gx = np.sum(arrayX)
            gy = np.sum(arrayY)
            gradient_magnituide = m.sqrt(gx ** 2 + gy ** 2)
            if gradient_magnituide < THRESHOLD:
                gradient_magnituide = 0
            else:
                gradient_magnituide = 255
            row.append(gradient_magnituide)
        output.append(row)
    return(np.array(output).astype(np.uint8))


def main():
    path = input("Enter the path to the image file: ")
    img_array = (load_img(path)*255).astype(np.uint8)

    test_image = rgb_to_grayscale(img_array)
    test_image = gaussian_filter(test_image, 1)
    test_image = sobel_filter(test_image)

    plt.imshow(test_image, cmap='gray', vmin=0, vmax=255)
    plt.show()
    return (0)

if __name__ == "__main__":
    main()
