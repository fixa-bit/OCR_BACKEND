import re
from flask import Flask, request
import pathlib
from pdf2image import convert_from_path
import io
import os
from PIL import Image
from werkzeug.utils import secure_filename
import pytesseract
from flask import Flask, render_template, request,jsonify
import json
from flask import Flask, request
from PIL import Image
import io
import ocr_core
from transform import four_point_transform,enhance
from skimage.filters import threshold_local
import cv2
import imutils
import pytesseract
import numpy as np 
import sys
import logging
import cv2
import imutils
import numpy as np 
from process_id import get_info,pre_process
import numpy as np
import difflib
import re
from collections import Counter
from flask_cors import CORS
global response

app = Flask(__name__)
CORS(app)
all_wordsx = np.load('all_words.npy')
scoresx = np.load('scores.npy')
response = ''
all_words = all_wordsx.tolist()
scores = scoresx.tolist()


@app.route('/extract_pdf', methods=['POST','GET'])
def extract_pdf():
    if 'file' not in request.files:
        return 'No file provided', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No file selected', 400
    page_number = request.form.get('page_number') 
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(filename)
        if page_number:
            extracted_text = extract_text_from_page(filename, int(page_number))
        else:
            extracted_text = extract_text_from_whole_file(filename)
        os.remove(filename)
        return extracted_text
    
    return 'Invalid file format'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def process_pdf_file(filename):
    # Convert PDF to images
    images = convert_pdf_to_images(filename)
    
    extracted_text = ''
    
    # Perform OCR on each image
    for image in images:
        text = perform_ocr(image)
        extracted_text += text + '\n'
    
    return extracted_text

def convert_pdf_to_images(pdf_path,page_number=None):
    if page_number:
        images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    else:
        images = convert_from_path(pdf_path)
    return images
    
def extract_text_from_page(pdf_path, page_number):
    images = convert_pdf_to_images(pdf_path, page_number)
    extracted_text = process_images(images)
    if extracted_text:
        return extracted_text  
    else :
        return 'Text not found'

def extract_text_from_whole_file(pdf_path):
    images = convert_pdf_to_images(pdf_path)
    extracted_text = process_images(images)
    if extracted_text:
        return extracted_text  
    else :
        return 'Text not found'

def process_images(images):
    extracted_text = ''
    for image in images:
        text = perform_ocr(image)
        extracted_text += text + '\n'
    return extracted_text  
 
    

def perform_ocr(image):
    # Apply OCR using Tesseract
    config = ('-l 4M_DER_69   --oem 1 --psm 6')
    text = pytesseract.image_to_string(image,config=config)
    print(text)
    return text


# @app.route('/extract_pdf', methods=['POST'])
# def extract_pdf():
#     if 'file' not in request.files:
#         return 'No file provided', 400
    
#     file = request.files['file']
    
#     if file.filename == '':
#         return 'No file selected', 400
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(filename)
#         extracted_text = process_pdf_file(filename)
#         os.remove(filename)
#         return extracted_text
    
#     return 'Invalid file format'

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

# def process_pdf_file(filename):
#     # Convert PDF to images
#     images = convert_pdf_to_images(filename)
    
#     extracted_text = ''
    
#     # Perform OCR on each image
#     for image in images:
#         text = perform_ocr(image)
#         extracted_text += text + '\n'
    
#     return extracted_text

# def convert_pdf_to_images(pdf_path):
#    images = convert_from_path(pdf_path)
#    return images
    
  
    
    

# def perform_ocr(image):
#     # Apply OCR using Tesseract
#     config = ('-l 4M_DER_69 --oem 1 --psm 6')
#     text = pytesseract.image_to_string(image,config=config)
#     print(text)
#     return text


@app.route('/extract_text', methods=['POST'])
def extract_text():
    file = request.files['image']
    #image_file = Image.open(file.stream)
    #selected_file=image.adjust_light_file_selection(file)
    text=ocr_core.pipe(file)
    if text:
        return text  
    else :
        return 'Text not found'
   #text = pytesseract.image_to_string(image,config=config)
    

    
    
@app.route('/extract_image', methods=['POST'])
def extract_image():
    image1= request.files['image1']
    image_file = Image.open(image1.stream)
    config = ('-l 4M_DER_69 --oem 1 --psm 6')
    text = pytesseract.image_to_string(image_file,config=config)
    if text:
        return text  
    else :
        return 'Text not found'



@app.route('/extract_id', methods=['POST'])    

def extract_id():
    if 'image_id' in request.files:
        image1 = request.files['image_id']
       
        img_final=get_transformed_image(image1)
        scanned=pre_process(img_final)
        info=get_info(scanned)
        # print(info)
        out_line=''
        for line in info:
            out_line=out_line+ '\n'+line
        if out_line:
            return out_line  
        else :
            return 'Text not found'
    
       
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
    




#@app.route('/')
@app.route('/name', methods = ['GET', 'POST'])
def hello_world():
   

    # checking the request type we get from the app
    if (request.method == 'POST'):
        request_data = request.data  # getting the response data
        request_data = json.loads(request_data.decode('utf-8'))  # converting it from json to key value pair
        inputx = request_data['name']
        print(inputx)




  

    def editreco3(input, original, val):
        trueoutcomes = difflib.get_close_matches(input, original, 1, 1)
        # print("input",input,"orgnal",original)
        if (len(trueoutcomes)):
            outcomes = input
        else:
            if (val == "delete"):  # single replacement for insert and delete

                if (len(original[0]) <= 2):
                    sim_index = 0.5
                elif (len(original[0]) <= 4):
                    sim_index = 0.6
                elif (len(original[0]) <= 5):

                    sim_index = 0.8
                else:
                    sim_index = 0.85
            if (val == "replace"):  # replace

                if (len(original[0]) <= 2):
                    sim_index = 0.5
                elif (len(original[0]) <= 3):
                    sim_index = 0.6
                elif (len(original[0]) <= 4):
                    sim_index = 0.7
                elif (len(original[0]) <= 6):

                    sim_index = 0.8
                else:
                    sim_index = 0.85

            if (val == "insert"):  # replacement for insert

                if (len(original[0]) <= 2):
                    sim_index = 0.5
                elif (len(original[0]) <= 4):
                    sim_index = 0.6
                elif (len(original[0]) <= 5):

                    sim_index = 0.8
                else:
                    sim_index = 0.85

            if (val == "swap"):

                if (len(original[0]) <= 2):
                    sim_index = 0.5
                elif (len(original[0]) <= 4):
                    sim_index = 0.7
                elif (len(original[0]) >= 5):

                    sim_index = 0.8  # for two saps use 0.7

            if (val == "edit2"):
                if (len(original[0]) <= 2):
                    sim_index = 0.3
                elif (len(original[0]) <= 3):
                    sim_index = 0.4
                elif (len(original[0]) <= 4):

                    sim_index = 0.8
                else:
                    sim_index = 0.6

            # print("big")
            outcomesx = difflib.get_close_matches(input, original, 1,
                                                  sim_index)  # input would be word selected after edit1 and original is word that was altered by edit1 function
            # outcomesmax=max(outcomes)
            if (len(outcomesx) > 0):
                outcomes = input
            else:
                outcomes = ""

        return outcomes

    def words(text):
        return re.findall(r'\w+', text.lower())

    # WORDS = Counter(words(open('big.txt').read()))
    # WORDS = Counter(words(open('ground_truth_record.txt','r').read()))
    WORDS = Counter(all_words)

    def P(word, N=sum(WORDS.values())):
        "Probability of `word`."
        return WORDS[word] / N

   

    def correction(word):

        scorewords1 = []
        scorewords2 = []
        spacescore = []
        candidate6xspace = (knownspace(editspace1(word)))  # space
        # print("space candidate",candidate6xspace)

        # index = all_words.index('کے')
        # wordlistx=wordlist+candidate6xspace

        all_optionsx = candidates(word)
        all_options = all_optionsx
        # print("all_optionslen ",len(all_options),len(all_optionsx))
        index_word = []
        scorefinal = []
        score_word = []
        sortedoutput = []
        for words in candidate6xspace:

            score1index = all_words.index(words.split(' ')[0])
            score2index = all_words.index(words.split(' ')[1])
            # print(score1index,score2index)

            if (len(words.split(' ')[0]) <= 3):
                scorewords1 = int(int(scores[score1index]) / 4)
                # print(str(words.split(' ')[0]),"is less then 3")
            else:
                scorewords1 = int(int(scores[score1index]))

            if (len(words.split(' ')[1]) <= 3):
                # print(str(words.split(' ')[1]),"is less then 3")
                scorewords2 = int(int(scores[score2index]) / 4)

            else:
                scorewords2 = int(int(scores[score2index]))

            newscorespace = int((scorewords1 + scorewords2) / 2)
            # print(newscorespace)
            # print("scores of new words after dividing with 4",str(scorewords1),str(scorewords2),"=",str(newscorespace))

            spacescore.append(str(newscorespace))
            # scorefinal=scores+spacescore

        for words2 in all_options:
            indexx = all_words.index(words2)
            index_word.append(str(indexx))
            newscore = scores[indexx]
            score_word.append(newscore)

        all_options = all_optionsx + candidate6xspace
        all_scores = score_word + spacescore

        # print(all_options,all_scores)

        # print(scorefinal)
        # scoresfinal=scores+scorex
        test_list1 = [s + ":" for s in all_options]
        res = [i + j for i, j in zip(test_list1, all_scores)]

        sortedlist = sorted(res, key=lambda e: int(e.split(':')[1]), reverse=True)
        # print('sortedlistxx',sortedlist)
        for items in sortedlist:
            sortednamesx = items.split(':')[0]
            sortedoutput.append(sortednamesx)
        # additional space information
        # print(candidate6xspace)
        # print(sortedoutput[:7])
        if(len(sortedoutput)<7):
            out_word=sortedoutput
        else:
            out_word=sortedoutput[:7]

        return out_word

    def candidates(word):
        "Generate possible spelling corrections for word."
        org = [word]
        # print("known words",known([word]),"edit1",len(known(edits1(word))),"edit2")
        candidate1 = list(known([word]))  # word already correctly spelled
        if (len(candidate1) != 0):
            # print("original",candidate1)
            finalcandidate = candidate1
        else:

            candidate2xdel = list(known(editdel1(word)))  # deleted
            candidate2xreplace = list(known(editreplace1(word)))  # deleted
            # print("candidate2xreplace",candidate2xreplace)
            candidate2xinsert = list(known(editinsert1(word)))  # deleted

            candidate3x = list(known(edits2(word)))  # twice
            # candidate4x=list([word])#unchanged bcz no option satisfied
            candidate5x = list(known(edittrans1(word)))

            candidate6xspace = (knownspace(editspace1(word)))  # space

            candidate2del = filter_word(candidate2xdel, org, "delete")
            candidate2replace = filter_word(candidate2xreplace, org, "replace")
            candidate2insert = filter_word(candidate2xinsert, org, "insert")
            candidate3 = filter_word(candidate3x, org, "edit2")

            candidate5 = filter_word(candidate5x, org, "swap")
            candidate6 = candidate6xspace
            finalcandidatex = candidate2del + candidate2insert + candidate2replace + candidate3 + candidate5  # + candidate6
            finalcandidate = list(set(finalcandidatex))
        return finalcandidate

    def known(wordss):
        "The subset of `words` that appear in the dictionary of WORDS."

        return set(w for w in wordss if w in WORDS)

    def knownspace(spaceset):
        "The subset of `words` that appear in the dictionary of WORDS."
        '''for item in spaceset:
          print(item[0],item[1])'''
        spacelist = []
        spaceoptions = set(item for item in spaceset if
                           (item[0] in WORDS and item[1] in WORDS) and len(item[1]) > 1 and len(item[0]) > 1)

        for item in spaceoptions:
            spacelist.append(item[0] + " " + item[1])

        return spacelist

    def edits1(word):
        "All edits that are one edit away from `word`."
        # letters    = 'abcdefghijklmnopqrstuvwxyz'
        # letters    = 'ا آ ب پ ت ٹ ث ج چ ح خ دڈذرڑزژس ش ص ض ط ظ ع غ ف ق ک گ ل م ن ں و ہ ھ ء ی ے'
        letters = 'اآبپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوہھءیے'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]

        return set(deletes + transposes + replaces + inserts)

    def editspace1(word):
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        return splits

    def edits2(word):
        "All edits that are two edits away from `word`."
        return (e2 for e1 in edits1(word) for e2 in edits1(e1))

    def editdel1(word):
        "All edits that are one edit away from `word`."
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        deletes = [L + R[1:] for L, R in splits if R]
        # print("del",len(deletes),deletes)

        return set(deletes)

    def edittrans1(word):
        "All edits that are one edit away from `word`."
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        # print("swaps",len(transposes),transposes)

        return set(transposes)

    def editreplace1(word):
        "All edits that are one edit away from `word`."
        letters = 'ا آ ب پ ت ٹ ث ج چ ح خ دڈذرڑزژس ش ص ض ط ظ ع غ ف ق ک گ ل م ن ں و ہ ھ ء ی ے'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        # print("replaces[0]",len(replaces),replaces)

        return set(replaces)

    def editinsert1(word):
        "All edits that are one edit away from `word`."
        letters = 'ا آ ب پ ت ٹ ث ج چ ح خ دڈذرڑزژس ش ص ض ط ظ ع غ ف ق ک گ ل م ن ں و ہ ھ ء ی ے'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        inserts = [L + c + R for L, R in splits for c in letters]
        # print("insert[0]",len(inserts),inserts[0])

        return set(inserts)

    def filter_word(word_list, original, val):

        ii = 0
        filtered_list = []
        '''if(len(word_list)==0):
          filtered_list=[]'''
        for elements in word_list:
            # print(elements)
            # editreco3(elements,original)
            filtered_listx = editreco3(elements, original, val)
            if (ii == 0):
                filtered_list = [filtered_listx]
                filtered_list = [i for i in filtered_list if i]
            else:
                filtered_list.append(filtered_listx)
                filtered_list = [i for i in filtered_list if i]
            ii = ii + 1

        return filtered_list

    xxx=correction(inputx)
    a = u', '.join(xxx)
    json_file = {}
    json_file['query'] = a
    return jsonify(json_file)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



# route and function to handle the upload page
@app.route('/webapp', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # check if there is a file in the request
        if 'file' not in request.files:
            return render_template('index.html', msg='No file selected')
        file = request.files['file']
        # if no file is selected
        if file.filename == '':
            return render_template('index.html', msg='No file selected')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # call the OCR function on it
            # extracted_text = ocr_core(file)

            config = ('-l 4M_DER_69   --oem 1 --psm 6')
            extracted_text = pytesseract.image_to_string(image,config=config)

            # extract the text and display it
            return render_template('index.html',
                                   msg='Successfully processed',
                                   extracted_text=extracted_text,
                                   #img_src=UPLOAD_FOLDER + file.filename)
                                   filename=filename)
    elif request.method == 'GET':
        return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)







if __name__ == '__main__':
  app.run(debug=True,host='0.0.0.0',port=5000)


