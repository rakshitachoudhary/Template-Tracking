import os
import cv2
import numpy as np
import copy

from numpy.core.fromnumeric import shape

def affineLK(img, template, x_start, x_end, y_start, y_end):
    p = np.zeros(6)
    p[0] += 1
    p[3] += 1
    y, x = template.shape
    sh = (img.shape[1], img.shape[0])
    for i in range(50):
        mat = np.array([[p[0],p[2],p[4]], [p[1], p[3], p[5]]])
        warp = cv2.warpAffine(img, mat, sh, flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP)
        k1 = np.array([[-1,0,1]])
        k2 = np.array([[-1],[0],[1]])
        warp_grad_x = cv2.filter2D(warp,cv2.CV_64F,k1)[y_start:y_end, x_start:x_end]
        warp_grad_x[warp_grad_x < 0] += 256
        warp_grad_y = cv2.filter2D(warp,cv2.CV_64F,k2)[y_start:y_end, x_start:x_end]
        warp_grad_y[warp_grad_y < 0] += 256
        dI = np.expand_dims((np.stack((warp_grad_x, warp_grad_y), 2)),2)
        warp = warp[y_start:y_end, x_start:x_end]
        diff = (warp - template).reshape((y, x, 1, 1))
        x_vec = np.array(range(x_start,x_end))
        y_vec = np.array(range(y_start, y_end))
        x_vec, y_vec = np.meshgrid(x_vec, y_vec)
        z = np.zeros(template.shape)
        o = np.ones(template.shape)
        jacobian = np.stack((np.stack((x_vec, z, y_vec, z, o, z), 2), np.stack((z, x_vec, z, y_vec, z, o), 2)), 2)
        dI_dW = np.matmul(dI, jacobian)
        dI_dW_trans = np.transpose(dI_dW, (0, 1, 3, 2))
        H = (np.matmul(dI_dW_trans, dI_dW).sum(0)).sum(0)
        S = ((dI_dW_trans* diff).sum(0)).sum(0)
        dP = np.matmul(np.linalg.pinv(H), S)
        for j in range(6):
            p[j] += dP[j]
        norm = np.linalg.norm(dP)
        if (norm < 0.01):
            break
    return p

def projectiveLK(img, template, x_start, x_end, y_start, y_end):
    p = np.zeros(9)
    p[0] += 1
    p[4] += 1
    p[8] += 1
    y, x = template.shape
    sh = (img.shape[1], img.shape[0])
    for i in range(50):
        mat = np.array([[p[0],p[1],p[2]], [p[3], p[4], p[5]], [p[6], p[7], p[8]]])
        warp = cv2.warpPerspective(img, mat, sh, flags=cv2.INTER_LINEAR)
        k1 = np.array([[-1,0,1]])
        k2 = np.array([[-1],[0],[1]])
        warp_grad_x = cv2.filter2D(warp,cv2.CV_64F,k1)[y_start:y_end, x_start:x_end]
        warp_grad_x[warp_grad_x < 0] += 256
        warp_grad_y = cv2.filter2D(warp,cv2.CV_64F,k2)[y_start:y_end, x_start:x_end]
        warp_grad_y[warp_grad_y < 0] += 256
        dI = np.expand_dims((np.stack((warp_grad_x, warp_grad_y), 2)),2)
        warp = warp[y_start:y_end, x_start:x_end]
        diff = (warp - template).reshape((y, x, 1, 1))
        x_vec = np.array(range(x_start,x_end))
        y_vec = np.array(range(y_start, y_end))
        x_vec, y_vec = np.meshgrid(x_vec, y_vec)
        z = np.zeros(template.shape)
        o = np.ones(template.shape)
        j11 = x_vec/(x_vec*p[6] + y_vec*p[7] + p[8])
        j12 = y_vec/(x_vec*p[6] + y_vec*p[7] + p[8])
        j13 = o/(x_vec*p[6] + y_vec*p[7] + p[8])
        j17 = -1*x_vec*(x_vec*p[0] + y_vec*p[1] + p[2])/((x_vec*p[6] + y_vec*p[7] + p[8])*(x_vec*p[6] + y_vec*p[7] + p[8]))
        j18 = -1*y_vec*(x_vec*p[0] + y_vec*p[1] + p[2])/((x_vec*p[6] + y_vec*p[7] + p[8])*(x_vec*p[6] + y_vec*p[7] + p[8]))
        j19 = -1*(x_vec*p[0] + y_vec*p[1] + p[2])/((x_vec*p[6] + y_vec*p[7] + p[8])*(x_vec*p[6] + y_vec*p[7] + p[8]))
        j27 = -1*x_vec*(x_vec*p[3] + y_vec*p[4] + p[5])/((x_vec*p[6] + y_vec*p[7] + p[8])*(x_vec*p[6] + y_vec*p[7] + p[8]))
        j28 = -1*y_vec*(x_vec*p[3] + y_vec*p[4] + p[5])/((x_vec*p[6] + y_vec*p[7] + p[8])*(x_vec*p[6] + y_vec*p[7] + p[8]))
        j29 = -1*(x_vec*p[3] + y_vec*p[4] + p[5])/((x_vec*p[6] + y_vec*p[7] + p[8])*(x_vec*p[6] + y_vec*p[7] + p[8]))
        jacobian = np.stack((np.stack((j11, j12, j13, z, z, z, j17, j18, j19), 2), np.stack((z, z, z, j11, j12, j13, j27, j28, j29), 2)), 2)
        dI_dW = np.matmul(dI, jacobian)
        dI_dW_trans = np.transpose(dI_dW, (0, 1, 3, 2))
        H = (np.matmul(dI_dW_trans, dI_dW).sum(0)).sum(0)
        S = ((dI_dW_trans* diff).sum(0)).sum(0)
        dP = np.matmul(np.linalg.pinv(H), S)
        for j in range(9):
            p[j] += dP[j]
        norm = np.linalg.norm(dP)
        if (norm < 0.01):
            break
    return p

def translationLK(img, template, x_start, x_end, y_start, y_end):
    p = np.zeros(2)
    y, x = template.shape
    sh = (img.shape[1], img.shape[0])
    for i in range(50):
        mat = np.array([[1,0,p[0]], [0, 1, p[1]]])
        warp = cv2.warpAffine(img, mat, sh, flags=cv2.INTER_LINEAR)
        k1 = np.array([[-1,0,1]])
        k2 = np.array([[-1],[0],[1]])
        warp_grad_x = cv2.filter2D(warp,cv2.CV_64F,k1)[y_start:y_end, x_start:x_end]
        warp_grad_x[warp_grad_x < 0] += 256
        warp_grad_y = cv2.filter2D(warp,cv2.CV_64F,k2)[y_start:y_end, x_start:x_end]
        warp_grad_y[warp_grad_y < 0] += 256
        dI = np.expand_dims((np.stack((warp_grad_x, warp_grad_y), 2)),2)
        warp = warp[y_start:y_end, x_start:x_end]
        diff = (warp - template).reshape((y, x, 1, 1))
        z = np.zeros(template.shape)
        o = np.ones(template.shape)
        jacobian = np.stack((np.stack((o, z), 2), np.stack((z, o), 2)), 2)
        dI_dW = np.matmul(dI, jacobian)
        dI_dW_trans = np.transpose(dI_dW, (0, 1, 3, 2))
        H = (np.matmul(dI_dW_trans, dI_dW).sum(0)).sum(0)
        S = ((dI_dW_trans* diff).sum(0)).sum(0)
        dP = np.matmul(np.linalg.pinv(H), S)
        for j in range(2):
            p[j] += dP[j]
        norm = np.linalg.norm(dP)
        if (norm < 0.01):
            break
    return p


def LKTracker(input_path, method_category):
    image = cv2.imread(input_path + "img/0001.jpg",0)
    f = open(input_path + "groundtruth_rect.txt", 'r')
    line = f.readline()
    words = line.split(",")
    x_start = int(words[0])
    y_start = int(words[1])
    x_end = x_start + int(words[2])
    y_end = y_start + int(words[3])
    temp = image[y_start:y_end, x_start:x_end]
    cv2.imshow("Template", temp)
    cv2.waitKey(0)

    point1 = np.array([x_start, y_start, 1])
    point2 = np.array([x_end, y_end, 1])
    point1_new = np.array([x_start, y_start])
    point2_new = np.array([x_end, y_end])
    print(point1, point2)
    files = os.listdir(input_path + "img")
    files.sort()
    l = len(files)

    open(input_path + "result_LK_" + method_category + ".txt", 'w').close()
    with open(input_path + "result_LK_" + method_category + ".txt", 'a') as out:
            out.write(str(x_start) + ',' + str(y_start) + ',' + str(x_end-x_start) + ',' + str(y_end-y_start)  + '\n')

    for f in range(1,l):
        frame = cv2.imread(os.path.join(input_path, "img/", files[f]),0)

        #affine
        if (method_category == "affine"):
            p = affineLK(frame, temp, point1[0], point2[0], point1[1], point2[1])
            mat = np.array([[p[0],p[2],p[4]], [p[1], p[3], p[5]]])
            point1_new = np.matmul(mat,point1).astype(int)
            point2_new = np.matmul(mat,point2).astype(int)

        #projective
        elif (method_category == "projective"):
            p = projectiveLK(frame, temp, point1[0], point2[0], point1[1], point2[1])
            point1_new = np.array([(p[0]*point1[0] + p[1]*point1[1] + p[2])/(p[6]*point1[0] + p[7]*point1[1] + p[8]), (p[3]*point1[0] + p[4]*point1[1] + p[5])/(p[6]*point1[0] + p[7]*point1[1] + p[8])]).astype(int)
            point2_new = np.array([(p[0]*point2[0] + p[1]*point2[1] + p[2])/(p[6]*point2[0] + p[7]*point2[1] + p[8]), (p[3]*point2[0] + p[4]*point2[1] + p[5])/(p[6]*point2[0] + p[7]*point2[1] + p[8])]).astype(int)

            if (point1_new[0] < 0):
                point1_new[0] = 0
            if (point1_new[1] < 0):
                point1_new[1] = 0
            if (point2_new[0] > image.shape[1]):
                point2_new[0] = image.shape[1]
            if (point2_new[1] > image.shape[0]):
                point2_new[1] = image.shape[0]
            if (point1_new[0] > image.shape[1]):
                point1_new[0] = image.shape[1]
            if (point1_new[1] > image.shape[0]):
                point1_new[1] = image.shape[0]
            if (point2_new[0] < 0):
                point2_new[0] = 0
            if (point2_new[1] < 0):
                point2_new[1] = 0

        #translation
        else:
            p = translationLK(frame, temp, point1[0], point2[0], point1[1], point2[1])
            point1_new = np.array([point1[0] + p[0], point1[1] + p[1]]).astype(int)
            point2_new = np.array([point2[0] + p[0], point2[1] + p[1]]).astype(int)

        xx_start=point1_new[0]
        xx_end=point2_new[0]
        yy_start=point1_new[1]
        yy_end=point2_new[1]
        with open(input_path + "result_LK_" + method_category + ".txt", 'a') as out:
            out.write(str(xx_start) + ',' + str(yy_start) + ',' + str(xx_end-xx_start) + ',' + str(yy_end-yy_start)  + '\n')
        print(point1_new, point2_new)

        #consecutive templates
        # point1 = np.array([point1_new[0], point1_new[1], 1])
        # point2 = np.array([point2_new[0], point2_new[1], 1])
        # cv2.imshow('Temp Image', frame[point1[1]:point2[1], point1[0]:point2[0]])
        # temp =  frame[point1[1]:point2[1], point1[0]:point2[0]]

        cv2.rectangle(frame, tuple(point1_new), tuple(point2_new), (255, 0, 0), 3)
        cv2.imshow('Tracked Image', frame)
        cv2.imwrite("dem1/"+files[f], frame)
        cv2.waitKey(10)
        
        
    cv2.destroyAllWindows()