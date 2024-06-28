import cv2
import imutils
import numpy as np 
from process_id import get_info,pre_process
from flask import Flask, request
import json
from PIL import Image

app = Flask(__name__)


@app.route('/extract_id', methods=['POST'])    

def main():
    if 'image_id' in request.files:
        image1 = request.files['image_id']
       
        img_final=get_transformed_image(image1)
        scanned=pre_process(img_final)
        info=get_info(scanned)
        return info
    else:
        return "No image provided", 400

def keypoint_matching(base,in_image):
    baseImg=base
    baseH, baseW, baseC = baseImg.shape
    base1_np=np.asarray(baseImg)

    #Init orb, keypoints detection on base Image
    orb = cv2.ORB_create(10000)

    kp, des = orb.detectAndCompute(baseImg, None)
    # imgKp = cv2.drawKeypoints(baseImg,kp, None)

    # display_img(imgKp)
    # cv2.imshow("base_keypoints", imgKp )
    # cv2.waitKey(0)
    PER_MATCH = 0.25

    #Detect keypoint on in_image
    kp1, des1 = orb.detectAndCompute(in_image, None)

    #Init BF Matcher, find the matches points of two images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(bf.match(des1, des))

    #Select top 30% best matcher 
    matches.sort(key=lambda x: x.distance)
    best_matches = matches[:int(len(matches)*PER_MATCH)]

    #Show match img  
    # imgMatch = cv2.drawMatches(in_image, kp1, baseImg, kp, best_matches,None, flags=2)
    # display_img(imgMatch)

    # cv2.imshow("new_keypoints", imutils.resize(imgMatch ,height=650))
    # cv2.waitKey(0)
    #Init source points and destination points for findHomography function.
    srcPoints = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1,1,2)
    dstPoints = np.float32([kp[m.trainIdx].pt for m in best_matches]).reshape(-1,1,2)
    #Find Homography of two images
    matrix_relationship, _ = cv2.findHomography(srcPoints, dstPoints,cv2.RANSAC, 5.0)

    #Transform the image to have the same structure as the base image
    img_final = cv2.warpPerspective(in_image, matrix_relationship, (baseW, baseH))

    # display_img(img_final)
    np_1=np.asarray(img_final)
    # dist = np.sum(np.abs(base1_np-np_1))
    dist= np.sum(np.abs(np.subtract(base1_np,np_1,dtype=np.float64))) / base1_np.shape[0]

    return img_final , dist




def get_transformed_image(input_image):
    try:
      in_image = cv2.imdecode(np.frombuffer(input_image.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
     

    except Exception as e:
        return str(e)
    # in_image               = cv2.imread(input_image)
    base_front             = cv2.imread("samples_id/ID_front.jpeg")
    base_back              = cv2.imread("samples_id/ID_back.jpeg")
  
    image_front,dist_front = keypoint_matching(base_front,in_image)
    image_back ,dist_back  = keypoint_matching(base_back,in_image)
    # print(dist_front,dist_back)
    if dist_front < dist_back:
        return image_front
    else:
        return image_back
    

   

if __name__ == '__main__':
  app.run(debug=True,host='0.0.0.0',port=8000)