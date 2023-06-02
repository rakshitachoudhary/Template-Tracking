import cv2
import numpy as np


points = np.zeros((2,2),np.int)
counter = 0

def mousePoints(event,x,y,flags,params):
    global counter
    if counter < 2 and event == cv2.EVENT_LBUTTONDOWN:
        points[counter] = x,y
        counter = counter + 1

temp = np.array([0])
cap = cv2.VideoCapture(0)
points_selected = False
templates = []
while True:
    ret, frame = cap.read()
    cv2.imshow("Original Image ", frame)
    key = cv2.waitKey(1)
    if key == 27: break
    
    while points_selected == False:
        for x in range (0,2):
            cv2.circle(frame, (points[x][0], points[x][1]), 3, (0, 0, 255), cv2.FILLED)
            
        cv2.imshow("Original Image ", frame)
        cv2.setMouseCallback("Original Image ", mousePoints)
        cv2.waitKey(1)

        if (counter >= 2): 
            points_selected = True
            print(points)
            for x in range (0,2):
                cv2.circle(frame, (points[x][0], points[x][1]), 3, (0, 0, 255), cv2.FILLED)
            start_x = points[0][0]
            start_y = points[0][1]
            end_x = points[1][0]
            end_y = points[1][1]
            temp = frame[start_y:end_y, start_x:end_x]
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
            # size9=(int(1.5*temp.shape[1]),int(1.5*temp.shape[0]))
            # temp9=cv2.resize(temp, size9)
            # size10=(int(1.6*temp.shape[1]),int(1.6*temp.shape[0]))
            # temp10=cv2.resize(temp, size10)
            # size11=(int(1.7*temp.shape[1]),int(1.7*temp.shape[0]))
            # temp11=cv2.resize(temp, size11)


            size12=(int(0.97*temp.shape[1]),int(0.97*temp.shape[0]))
            temp12=cv2.resize(temp, size12)
            size13=(int(0.93*temp.shape[1]),int(0.93*temp.shape[0]))
            temp13=cv2.resize(temp, size13)
            size14=(int(0.89*temp.shape[1]),int(0.89*temp.shape[0]))
            temp14=cv2.resize(temp, size14)
            size15=(int(0.85*temp.shape[1]),int(0.85*temp.shape[0]))
            temp15=cv2.resize(temp, size15)
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


            templates = [temp,temp1,temp2,temp3,temp4,temp5,temp6,temp7,temp8,temp12,temp13,temp14,temp15]
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 3)
            cv2.imshow("Original Image ", frame)
            cv2.waitKey(0)
            break

    curr_max = 0
    index = 0     
    for i in range(len(templates)):
        match= cv2.matchTemplate(frame,templates[i],cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, _ = cv2.minMaxLoc(match)
        if maxVal > curr_max:
            curr_max = maxVal
            index = i
    match = cv2.matchTemplate(frame, templates[index], cv2.TM_CCOEFF_NORMED)
    (_, _, min_cord, max_cord) = cv2.minMaxLoc(match)
    (x_start, y_start) = max_cord
    x_end = x_start + templates[index].shape[1]
    y_end = y_start + templates[index].shape[0]
    
    cv2.rectangle(frame, max_cord, (x_end, y_end), (255, 0, 0), 3)
    cv2.imshow("Result", frame)
    cv2.waitKey(10)

cap.release()
cv2.destroyAllWindows()