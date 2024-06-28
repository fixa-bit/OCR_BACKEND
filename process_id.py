from skimage.filters import threshold_local
import cv2
import imutils
import pytesseract
import numpy as np 
import sys
import re
from fuzzywuzzy import fuzz

def find_gender(text_list):
    if len(text_list)>0:
        for line in text_list:
            if "F" in line:
                return "Female"
            elif "M" in line:
                return "Male"
            elif "X" in line:
                return "Transgender"
            else:
                pass


def closest_match(data):
    score=1
    match=""
    f=open("list_of_countries.txt","r")
    countries=f.readlines()
    for row in countries:
        score_tmp=fuzz.ratio(data,row.strip())
        if score_tmp>= score:
            score=score_tmp
            match=row.strip()
    return match

def find_numeric_line(text_list):
    output_line=""
    numeric=True
    if len(text_list)>0:
        for line in text_list:
            result = re.sub(' +',' ',re.sub(u'[\W\s]',' ',line))
            for char in result:
                if char == " ":
                    pass
                elif char.isnumeric() :
                    pass
                else:
                    numeric=False
            if numeric==True:
                output_line=output_line+str(line)
    return output_line

def find_alpha_line(text_list):
    output_line=""
    if len(text_list)>0:
        for line in text_list:
            result = re.sub(' +',' ',re.sub(u'[\W\s]',' ',line))
            alpha=True
            for char in result:
                if char == " ":
                    pass
                elif char.isnumeric() :
                    alpha=False
                else:
                    pass
            if alpha==True:
                output_line=output_line+str(line)
    return output_line

def get_text_eng(image):
    config = ('-l eng --oem 1 --psm 6')
    text = pytesseract.image_to_string(image ,config=config)
    text=text.replace("|","")
    text_list=text.splitlines()

    return text_list

def get_text_urdu(image):
    config = ('-l 4Mvf_DER_75 --oem 1 --psm 6')
    text = pytesseract.image_to_string(image ,config=config)
    text=text.replace("|","")
    text_list=text.splitlines()

    return text_list


def pre_process(camera_file):
    # img= cv2.imread(camera_file)
    img = imutils.resize(camera_file, height = 650)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noiseless_image_bw=gray
    noiseless_image_bw = cv2.fastNlMeansDenoising(gray, noiseless_image_bw ,5, 7, 21)
    T = threshold_local(noiseless_image_bw, 21, offset = 10, method = "gaussian")
    warped = (gray > T).astype("uint8") * 255
    std_scan=imutils.resize(warped, height = 650)


    return std_scan

def get_info(img):
    h,w=img.shape
    cropped_image_cat9 = img[h-int(h/7):h,int(w/4):w-int(w/4)]

    number_of_white_pix = np.sum(cropped_image_cat9 == 255)
    number_of_black_pix = np.sum(cropped_image_cat9 == 0)

    info=[]
    side="none"
    b_2_w_ratio=100
    if number_of_black_pix>0:
         b_2_w_ratio=number_of_white_pix/number_of_black_pix

    if b_2_w_ratio < 100:
        cropped_image_cat1 = img[1:120,225:875]
        text_list=get_text_eng(cropped_image_cat1)
        for line in text_list:
            if "PAKISTAN" in line:
                side="front"
    else:
        side="back"
    if side=="front":
        info.append('CNIC FRONT FACE DETECTED')
#NAME
        cropped_image_cat2 = img[175:225,int(w/4):int(w-w/4)]
        text_list=get_text_eng(cropped_image_cat2)
        line=find_alpha_line(text_list)
        info.append("Name : "+line)
#H/F NAME
        cropped_image_cat3 = img[290:330,int(w/4):int(w-w/4)]
        text_list=get_text_eng(cropped_image_cat3)
        line=find_alpha_line(text_list)
        info.append("Father's Name/Husband's Name: "+line)
#COUNTRY OF STAY
        cropped_image_cat4 = img[int(h/2+h/8):int(h/2+h/8+50),int(w/4+w/8):int(w-w/2)]
        text_list=get_text_eng(cropped_image_cat4)
        line=find_alpha_line(text_list)
        country=closest_match(line)
        info.append("Country of Stay: "+country)
#GENDER 
        cropped_image_cat13 = img[int(h/2+h/8):int(h/2+h/8+50),int(w/4):int(w/4+w/8)]
        text_list=get_text_eng(cropped_image_cat13)
        line=find_gender(text_list)
        info.append("Gender: "+str(line))
        
 
#CNIC
        cropped_image_cat5 = img[int(h/1.35):int(h/1.25),int(w/4):int(w-w/2)]
        text_list=get_text_eng(cropped_image_cat5)
        line=find_numeric_line(text_list)
        info.append("CNIC Number: "+str(line))
#DOB
        cropped_image_cat6 = img[int(h/1.35):int(h/1.25),int(w/2):int(w-w/3)]
        text_list=get_text_eng(cropped_image_cat6)
        line=find_numeric_line(text_list)
        info.append("Date of Birth: "+line)
#CARD ISSUE DATE
        cropped_image_cat7 = img[h-100:h-50,int(w/4):int(w-w/2)]
        text_list=get_text_eng(cropped_image_cat7)
        line=find_numeric_line(text_list)
        info.append("Card Issue Date: "+line)
#CARD EXPIRY DATE
        cropped_image_cat8 = img[h-100:h-50,int(w/2):int(w-w/3)]
        text_list=get_text_eng(cropped_image_cat8)
        line=find_numeric_line(text_list)
        info.append("Card Expiry Date: "+line)

    elif side == "back":

        info.append('CNIC BACK FACE DETECTED')
#line1
        cropped_image_cat10 = img[ 25:int(h/4),int(w/5):int(w-w/4)]
        text_list=get_text_urdu(cropped_image_cat10)
        line=find_alpha_line(text_list)
        info.append(line)
#line2
        cropped_image_cat11 = img[int(h/4):int(h/2),int(w/5):int(w-w/4)]
        text_list=get_text_urdu(cropped_image_cat11)
        line=find_alpha_line(text_list)
        info.append(line)
#CNIC
        cropped_image_cat12 = img[25:75,int(w-w/4):w]
        text_list=get_text_eng(cropped_image_cat12)
        line=find_numeric_line(text_list)
        info.append(line)
    
    else:
        info.append("invalid")
    return info

# def main(image):
#     scanned_image=pre_process(image)
#     info=get_info(scanned_image)
#     print(info)

# for i in range(1, len(sys.argv)):
#     print('argument:', i, 'value:', sys.argv[i])
#     main(sys.argv[i])

