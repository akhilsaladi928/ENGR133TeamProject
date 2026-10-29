"""
Course Number: ENGR 13300
Semester: Fall 2026

Description:
    Beginning of our team project; conversion of rgb to hsv files, etc.

Assignment Information:
    Assignment:     tp1 team 1
    Team ID:        LC4 - 18
    Author:         Oscar Anderson, ande2059@purdue.edu
    Date:           10/15/2026

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
from PIL import Image, ImageOps
import matplotlib.pyplot as plt


#load image function from py5 team task 2
def load_img(file_path):
    #load image and linearize it
    image = Image.open(file_path)
    arr = np.array(image)/255.0
    cond = 0.04045
    #check for alpha channel 
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3] #remove alpha channel
    return np.where(arr > cond, ((arr+0.055)/1.055)**2.4, arr/12.92)

#resize image function from py5 team task 3
def resize_image(original_image_array, new_size):
    #turn array into image using PIL's Image.fromarray() function
    image = Image.fromarray(original_image_array)
    newWidth, newHeight = new_size
    #resize the image using PIL's resize() function with bilinear resampling
    resized_image = image.resize((newWidth, newHeight), Image.Resampling.BILINEAR)
    #convert the resized image back to a numpy array
    resized_image_array = np.array(resized_image)
    return resized_image_array

#pad image function from py5 team task 3
def pad_image(resized_image_array, target_size):
    image = Image.fromarray(resized_image_array)
    padded_image = ImageOps.pad(image, target_size, method=Image.Resampling.BILINEAR, color=0)
    padded_image_array = np.array(padded_image)
    return padded_image_array


def rgb_to_hsv(r, g, b):
    red = float(r) / 255.0
    green = float(g) / 255.0
    blue = float(b) / 255.0

    c_max = max(red, green, blue)
    c_min = min(red, green, blue)
    delta = c_max - c_min

    if delta == 0:
        hue = 0.0
    elif c_max == red:
        hue = (60.0 * ((green - blue) / delta) + 360.0) % 360.0
    elif c_max == green:
        hue = (60.0 * ((blue - red) / delta) + 120.0) % 360.0
    else:
        hue = (60.0 * ((red - green) / delta) + 240.0) % 360.0

    if c_max == 0:
        sat = 0.0
    else:
        sat = delta / c_max

    value = c_max

    return hue * 255.0 / 360.0, sat * 255.0, value * 255.0

#finds unique colors, indices
#gets conversion of each unique pixel and recreates using the inverse map
#ok so this one's a lil complicated, found out how to do this after running index-for-index and
#it took like 5 minutes for a large image, this is a LOT faster
def convert_to_hsv(rgb_image):
    #get height and width for eventual output array
    h, w, _ = rgb_image.shape
    #essentially flattens it into a list of pixels, rather than 2d array
    flat = rgb_image.reshape(-1, 3)
    #get the VALUES and INDICES of each unique pixel color
    unique, indices = np.unique(flat, axis=0, return_inverse=True)
    #convert this into a flattened list of hsv pixels
    # run rgb_to_hsv on each unique color
    unique_hsv = np.empty(unique.shape, dtype=np.float64)
    for i in range(unique.shape[0]):
        unique_hsv[i] = rgb_to_hsv(int(unique[i, 0]), int(unique[i, 1]), int(unique[i, 2]))
    # cast to uint8 (truncates, as the assignment note describes)
    unique_hsv_u8 = unique_hsv.astype(np.uint8)
    #essentially un-flatten the image
    return unique_hsv_u8[indices.reshape(-1)].reshape(h, w, 3)

    


def main():
    path1 = input("Enter the path of the RGB image you want to convert to hsv: ")
    coords = input("Enter the coordinates of the pixel you want inspect: ").split(", ")
    x = int(coords[0])
    y = int(coords[1])
    img_array = (load_img(path1)*255).astype(np.uint8)
    #get og dimensions
    ogHeight, ogWidth = img_array.shape[0], img_array.shape[1]
    #tuple representing sizing target
    new_size = (100, 100)
    targetWidth, targetHeight = new_size
    # Calculate aspect ratios
    originalRatio = ogWidth / ogHeight
    targetRatio = targetWidth / targetHeight
    # Resize while maintaining aspect ratio
    if originalRatio >= targetRatio:
        newWidth = targetWidth
        newHeight = int(targetWidth / originalRatio)
    else:
        newHeight = targetHeight
        newWidth = int(targetHeight * originalRatio)
    new_size = (newWidth, newHeight)
    # Resize image
    resized_image_arr = resize_image(img_array, new_size)
    # Pad image to target dimensions
    padded_image = pad_image(
        resized_image_arr,
        (targetWidth, targetHeight)
    )
    #print rgb values at chosen coordinate
    r, g, b = resized_image_arr[y, x]
    print(f"RGB values of the ({x}, {y}) pixel: R={r}, G={g}, B={b}")
    #convert image to hsv
    hsv_image = convert_to_hsv(resized_image_arr)
    print(f"Converting {path1} to HSV...")
    #print hsv values at chosen coordinate
    hh, ss, vv = hsv_image[y, x]
    print(f"HSV values of the ({x}, {y}) pixel: H={hh}, S={ss}, V={vv}")
    plt.imshow(hsv_image)
    plt.show()



if __name__ == "__main__":
    main()
