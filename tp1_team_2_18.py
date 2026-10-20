"""
Course Number: ENGR 13300
Semester: e.g. Spring 2026

Description:
    Replace this line with a description of your program.

Assignment Information:
    Assignment:     tp1 team 2
    Team ID:        ### - ## (e.g. LC1 - 01; for section LC1, team 01)
    Author:         Name, login@purdue.edu
    Date:           e.g. 01/23/2026

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

""" Write any import statements here (and delete this line)."""
import numpy as np
from PIL import Image, ImageOps
import math

TESTIMAGE = [[0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 255, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]]

def create_mean_kernel(size):
    value = 1.0/(size**2)
    kernel = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(value)
        kernel.append(row)
    return np.array(kernel)

def pad_image(resized_image_array, target_size):
    #convert array back to image
    image = Image.fromarray(resized_image_array)

    #Pad the image using ImageOPS.expand
    padded_image = ImageOps.pad(image, target_size, method=Image.Resampling.BILINEAR, color=0)

    #Turn image back into an array
    padded_image_array = np.array(padded_image)

    return padded_image_array


def apply_kernel(image, kernel):
    image = np.array(image, dtype=np.uint8)
    kernel = np.array(kernel)
    height, width = image.shape
    kernelHeight, kernelWidth = kernel.shape
    padWith = kernelWidth // 2
    padHeight = kernelHeight // 2
    paddedImage = pad_image(image, (width + 2 * padWith, height + 2 * padHeight))
    #print(np.array2string(paddedImage))
    output = np.zeros(image.shape, dtype=float)

    for i in range(height):
        for j in range(width):
            window = paddedImage[i:i+kernelHeight, j:j+kernelWidth]
            output[i, j] = np.sum(window * kernel)

    return output

def main():
    size = int(input("Enter kernel size: "))
    print(f"Kernel size: {size}x{size}")
    kernal = (create_mean_kernel(size))
    sum = 0
    for i in range(size):
        for j in range(size):
            sum += kernal[i][j]
    print(f"Kernel sum: {sum:.6f}")
    print("Kernel:")
    print(np.array2string(kernal, precision=6))
    print("Filtered image:")
    filteredImage = apply_kernel(TESTIMAGE, kernal)
    print(np.array2string(filteredImage, precision=6))


if __name__ == "__main__":
    main()
