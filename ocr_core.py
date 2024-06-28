from PIL import ImageOps,Image
from PIL import ImageEnhance
import pytesseract
import cv2
import numpy as np
from skimage import io
from skimage.transform import rotate
from skimage import io
from skimage.color import rgb2gray
from deskew import determine_skew
import os

def Image_Enhance(filename):
    #selected image convertes to grayscale
    image = Image.open(filename).convert('L') 
    print(image.format)
    ## Enhance Brightness
    curr_bri = ImageEnhance.Brightness(image)
    new_bri = 0.5
    img_brightened = curr_bri.enhance(new_bri)
    # Enhance Contrast
    curr_con = ImageEnhance.Contrast(img_brightened)
    new_con = 1.5
    img_contrasted = curr_con.enhance(new_con)
    #Enhance Sharpness
    curr_sharp = ImageEnhance.Sharpness(img_contrasted)
    new_sharp = 1.5
    img_sharped = curr_sharp.enhance(new_sharp)
    return img_sharped

def Image_Binarize(filename):
    img1=np.array(filename)
    ret2,th2 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    out_image=Image.fromarray(th2)
    return out_image

def erode_dilate(img, kernel_size =1):
    img=np.array(img)
    kernel = np.ones((kernel_size,kernel_size), np.uint8) 
    img_dilation = cv2.dilate(img, kernel, iterations=2)
    img_erosion=cv2.erode(img, kernel, iterations=2)
    out_image=Image.fromarray(img_erosion)
    return out_image

def Image_Denoise(filename):
    img1=np.array(filename)
    denoised_image=cv2.fastNlMeansDenoising(img1)
    out_image=Image.fromarray(denoised_image)
    return out_image

def deskew(img):
    img=np.array(img)
    angle = determine_skew(img)
    rotated = rotate(img, angle, resize=True) * 255
    out_image=Image.fromarray(rotated)
    if out_image.mode != 'L':
        out_image = out_image.convert('L')
    return out_image


def invert_image(img):
    image_array=np.array(img)
    number_of_white_pix = np.sum(image_array == 255) 
    number_of_black_pix = np.sum(image_array == 0) 
    if number_of_black_pix > number_of_white_pix:
        inverted_image = ImageOps.invert(img)
        #out_image=Image.fromarray(inverted_image)
        out_image=inverted_image
    else:
        out_image=img
    return out_image

def ocr_core(filename,config):
    """
    This function will handle the core OCR processing of images.
    """
    # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    text = pytesseract.image_to_string(filename,config=config)
    return text


def pipe(image_selected):
    #st.image("pdf_images"+"/"+str(image_selected))
    image_enhanced=Image_Enhance(image_selected)
    #Applying binarization
    image_binarize=Image_Binarize(image_enhanced) 
    #Applying Invertion
    image_invert=invert_image(image_binarize)
    #Applying erosion dilation
    image_erode_dilate=erode_dilate(image_invert)
    #Removing Noise
    image_denoised=Image_Denoise(image_erode_dilate)
    #Deskewing Image
    image_deskew=deskew(image_denoised) 
    #image_deskew.show()
    #Rotating Image
    config = ('-l 4M_DER_69  --oem 1 --psm 6')
    text_1=ocr_core(image_deskew,config)
    return  text_1


            
  
