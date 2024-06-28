
from PIL import Image
import ocr_core 
import os
import tempfile
from pathlib import Path
import base64
import adjust_light as ad
from PIL import Image
import numpy as np
import cv2

def adjust_light_file_selection(image_selected):
    PATH="./adjust_image/"
    FILE_NAME=str("image")+"_"+'1.jpg'
    c = 0.4                   # in the range[0, 1] recommend 0.2-0.3
    bl = 260 
    im = Image.open(image_selected).convert('L')
    im=np.array(im)
    #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = im.astype(np.float32)
    #st.image(gray, clamp=True)
    width = gray.shape[1]
    height = gray.shape[0]
    hp = ad.get_hist(im)
    sqrt_hw = np.sqrt(height * width)
    hr = ad.get_hr(hp, sqrt_hw)
    cei = ad.get_CEI(gray, hr, c)
    m1 = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape((3,3))
    m2 = np.array([-2,-1,0,-1,0,1,0,1,2]).reshape((3,3))
    m3 = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape((3,3))
    m4 = np.array([0,1,2,-1,0,1,-2,-1,0]).reshape((3,3))
    eg1 = np.abs(cv2.filter2D(gray, -1, m1))
    eg2 = np.abs(cv2.filter2D(gray, -1, m2))
    eg3 = np.abs(cv2.filter2D(gray, -1, m3))
    eg4 = np.abs(cv2.filter2D(gray, -1, m4))
    eg_avg = ad.scale((eg1 + eg2 + eg3 + eg4) / 4)
    bins_1 = np.arange(0, 265, 5) 
    eg_bin = ad.img_threshold(30, eg_avg,"H2H")
    bins_2 = np.arange(0, 301, 40)
    #threshold_c = 255 - get_th2(cei, bins_2)
    cei_bin = ad.img_threshold(60, cei, "H2L")#threshold is hard coded to 60 (based 
                                        #on the paper). Uncomment above to replace
    #cv2.imwrite(FILE_NAME + "_CeiBin" + FORMAT, cei_bin)
    tli = ad.merge(eg_bin, cei_bin)
    #cv2.imwrite(1 + "_TLI" + FORMAT, tli)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(tli,kernel,iterations = 1)
    int_img = np.array(cei)
    ratio = int(width / 20)
    for y in range(width):
        if y % ratio == 0 :
            print(int(y / width * 100), "%")
        for x in range(height):
            if erosion[x][y] == 0:
                x = ad.set_intp_img(int_img, x, y, erosion, cei)
    mean_filter = 1 / 121 * np.ones((11,11), np.uint8)
    ldi = cv2.filter2D(ad.scale(int_img), -1, mean_filter)
    result = np.divide(cei, ldi) * bl
    result[np.where(erosion != 0)] *= 1.5
    data = Image.fromarray(result)
    result=cv2.imwrite(os.path.join(PATH,FILE_NAME), result) 
    image_file=os.path.join(PATH,FILE_NAME)
    return image_file



            


