"""
Course Number: ENGR 13300
Semester: e.g. Spring 2026

Description:
    Write two functions, one to build a Gaussian kernel and one to apply it, then use them to blur a user-selected grayscale image

Assignment Information:
    Assignment:     tp1 team 3
    Team ID:        LC4 - 18 (e.g. LC1 - 01; for section LC1, team 01)
    Author:         Rohan Ravuka, ravular@purdue.edu
    Date:           10/20/26

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
from PIL import Image
import math
import numpy as np

def create_gaussian_kernel(sigma):
    if sigma <= 0:
        raise ValueError("Sigma must be a positive number.")

    # Determine the kernel size so that it covers at least 3 standard deviations from the mean.
    radius = math.ceil(3 * sigma)
    kernel_size = 2 * radius + 1

    # Create coordinates for a square 2D kernel so that the center of the kernel is at (0, 0).
    coordinates = np.arange(-radius, radius + 1, dtype=float)
    xx, yy = np.meshgrid(coordinates, coordinates)

    # Create and normalize the 2D Gaussian kernel so that the sum of all elements equals 1.
    kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    kernel = kernel.astype(float)

    #normalize the kernel so that the sum of all elements equals 1
    kernel = kernel / np.sum(kernel)
   
    return kernel

def gaussian_filter(image, sigma):
    #building the kernel
    kernel = create_gaussian_kernel(sigma)

    #padding so that the kernel can be applied to the edges of the image
    radius = math.ceil(3 * sigma)
    kernel_size = 2 * radius + 1
    pad_width = radius

    #zero-padding so that the kernel can be applied to the edges of the image
    padded_image = np.pad(image, pad_width=int(pad_width), mode='constant', constant_values=0)

    #ouput image so that the original image is not modified
    output_image = np.zeros_like(image, dtype=float)

    #slide the kernel over the image stride 1 so that the kernel is applied to every pixel in the image
    img_h, img_w = image.shape

    for i in range(img_h):  #iterating through matrix rows so that the kernel can be applied to every pixel in the image

        for j in range(img_w):  #iterating through each row so that the kernel can be applied to every pixel in the image

            window = padded_image[i:i + kernel_size, j:j + kernel_size]

            output_image[i, j] = np.sum(window * kernel)

    output_image[np.isclose(output_image, 255, atol=1e-10)] = 255
    return np.clip(output_image, 0, 255).astype(np.uint8)


def rgb_to_grayscale(rgb_image):    #copied from other assignment
    #changing from rgb to grayscale
    weights = np.array([0.2126, 0.7152, 0.0722])
    return np.dot(rgb_image[:, :, :3], weights)



def main():
    #prompting for image path so that the user can select an image to blur
    img_path = (input("Enter the path to the image file: "))
    #converting image into an array so that it can be processed
    img_array = np.array(Image.open(img_path).convert("RGB"))
    #converting image into grayscale so that the gaussian filter can be applied to it
    img_grayscale = rgb_to_grayscale(img_array)

    #calling gaussian filter so that the image can be blurred
    blurred_image = gaussian_filter(img_grayscale, 1)
    blurred_image = np.clip(blurred_image, 0, 255).astype(np.uint8)

    #saving image so that the user can see the blurred image
    Image.fromarray(blurred_image).save("blurred.png")

if __name__ == "__main__":
    main()