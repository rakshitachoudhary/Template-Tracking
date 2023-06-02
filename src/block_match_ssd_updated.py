import os
import cv2
import numpy as np

def block_match_ssd(input_path):
    image = cv2.imread(input_path+"img/0001.jpg",0)
    f = open(input_path+"groundtruth_rect.txt", 'r')
    line = f.readline()
    words = line.split(",")
    x_start = int(words[0])
    y_start = int(words[1])
    x_end = x_start + int(words[2])
    y_end = y_start + int(words[3])
    temp = image[y_start:y_end, x_start:x_end]

    size1=(int(1.04*temp.shape[1]),int(1.04*temp.shape[0]))
    temp1=cv2.resize(temp, size1)
    size2=(int(1.08*temp.shape[1]),int(1.08*temp.shape[0]))
    temp2=cv2.resize(temp, size2)
    size3=(int(1.12*temp.shape[1]),int(1.12*temp.shape[0]))
    temp3=cv2.resize(temp, size3)
    size4=(int(1.16*temp.shape[1]),int(1.16*temp.shape[0]))
    temp4=cv2.resize(temp, size4)
    size5=(int(1.2*temp.shape[1]),int(1.2*temp.shape[0]))
    temp5=cv2.resize(temp, size5)
    size6=(int(1.24*temp.shape[1]),int(1.24*temp.shape[0]))
    temp6=cv2.resize(temp, size6)
    size7=(int(1.3*temp.shape[1]),int(1.3*temp.shape[0]))
    temp7=cv2.resize(temp, size7)
    size8=(int(1.4*temp.shape[1]),int(1.4*temp.shape[0]))
    temp8=cv2.resize(temp, size8)
    size9=(int(1.5*temp.shape[1]),int(1.5*temp.shape[0]))
    temp9=cv2.resize(temp, size9)
    size10=(int(1.6*temp.shape[1]),int(1.6*temp.shape[0]))
    temp10=cv2.resize(temp, size10)
    size11=(int(1.7*temp.shape[1]),int(1.7*temp.shape[0]))
    temp11=cv2.resize(temp, size11)


    # size12=(int(0.97*temp.shape[1]),int(0.97*temp.shape[0]))
    # temp12=cv2.resize(temp, size12)
    # size13=(int(0.93*temp.shape[1]),int(0.93*temp.shape[0]))
    # temp13=cv2.resize(temp, size13)
    # size14=(int(0.89*temp.shape[1]),int(0.89*temp.shape[0]))
    # temp14=cv2.resize(temp, size14)
    # size15=(int(0.85*temp.shape[1]),int(0.85*temp.shape[0]))
    # temp15=cv2.resize(temp, size15)
    # size16=(int(0.81*temp.shape[1]),int(0.81*temp.shape[0]))
    # temp16=cv2.resize(temp, size16)
    # size17=(int(0.76*temp.shape[1]),int(0.76*temp.shape[0]))
    # temp17=cv2.resize(temp, size17)
    # size18=(int(0.71*temp.shape[1]),int(0.71*temp.shape[0]))
    # temp18=cv2.resize(temp, size18)
    # size19=(int(0.61*temp.shape[1]),int(0.61*temp.shape[0]))
    # temp19=cv2.resize(temp, size19)
    # size20=(int(0.5*temp.shape[1]),int(0.5*temp.shape[0]))
    # temp20=cv2.resize(temp, size20)


    templates = [temp,temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp9,temp10,temp11]



    files = os.listdir(input_path+"img")
    files.sort()
    l = len(files)

    open(input_path+"result_ssd.txt", 'w').close()
    with open(input_path+"result_ssd.txt", 'a') as out:
            out.write(str(x_start) + ',' + str(y_start) + ',' + str(x_end-x_start) + ',' + str(y_end-y_start)  + '\n')
        
    for f in range(1,l):
        print(f)
        frame = cv2.imread(os.path.join(input_path+"img/", files[f]),0)

        curr_min = float('INF')
        index = 0     
        for i in range(len(templates)):
            match= cv2.matchTemplate(frame,templates[i],cv2.TM_SQDIFF)
            minval, _, _, _ = cv2.minMaxLoc(match)
            if minval < curr_min:
                curr_min = minval
                index = i
        match = cv2.matchTemplate(frame, templates[index], cv2.TM_SQDIFF)
        (_, _, min_cord, max_cord) = cv2.minMaxLoc(match)
        (x_start, y_start) = min_cord
        x_end = x_start + templates[index].shape[1]
        y_end = y_start + templates[index].shape[0]
        
        with open(input_path+"result_ssd.txt", 'a') as out:
            out.write(str(x_start) + ',' + str(y_start) + ',' + str(x_end-x_start) + ',' + str(y_end-y_start)  + '\n')
        
        cv2.rectangle(frame, min_cord, (x_end, y_end), (255, 0, 0), 3)
        cv2.imshow("Result", frame)
        cv2.waitKey(1)
        
    cv2.destroyAllWindows()