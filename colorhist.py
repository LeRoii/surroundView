from itertools import chain
import cv2
import os
import numpy as np
import datetime
 
class BrightnessBalance:
    def __init__(self):
        pass
 
    def arrayToHist(self,gray):
        '''
        计算灰度直方图，并归一化
        :param gray_path:
        :return:
        '''
        w,h = gray.shape
        hist = list(chain.from_iterable(cv2.calcHist([gray],[0],None,[256],[0,256])))
        hist = np.array(hist)/(w*h)
        return hist
 
    def histMatch(self, gray1, gray2):
        '''
        gray2向gray1校准
        :param image_path1:
        :param image_path2:
        :return:
        '''
        if not os.path.exists(image1_path) or not os.path.exists(image2_path):
            print('image1_path or image2_path is not exist!')
            return
        gray2Array = np.array(gray2)
        hist1 = self.arrayToHist(gray1)
        hist2 = self.arrayToHist(gray2Array)
        tmp1 = 0.0
        tmp2 = 0.0
        h_acc1 = hist1.copy()
        h_acc2 = hist2.copy()
        for i in range(256):
            tmp1 += hist1[i]
            tmp2 += hist2[i]
            h_acc1[i] = tmp1
            h_acc2[i] = tmp2
        M = np.zeros(256)
        for i in range(256):
            idx = 0
            minv = 1
            for j in range(0,len(h_acc1)):
                if (np.fabs(h_acc1[j] - h_acc2[i]) < minv):
                    minv = np.fabs(h_acc1[j] - h_acc2[i])
                    idx = int(j)
            M[i] = idx
        result = M[gray2Array]
        return result
 
    def colors_histMatch(self,image_path1,image_path2):
        if not os.path.exists(image1_path) or not os.path.exists(image2_path):
            print('image1_path or image2_path is not exist!')
            return
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        result1_chaneel_b = self.histMatch(img1[:, :, 0], img2[:, :, 0])
        result1_chaneel_g = self.histMatch(img1[:, :, 1], img2[:, :, 1])
        result1_chaneel_r = self.histMatch(img1[:, :, 2], img2[:, :, 2])
        result = cv2.merge([result1_chaneel_b,result1_chaneel_g,result1_chaneel_r])
        return result
 
if __name__ == '__main__':
    image1_path = '/space/code/multiview/v2/data/0407/back.png'
    image2_path = '/space/code/multiview/v2/data/0407/front.png'
    starttime = datetime.datetime.now()
    BB = BrightnessBalance()
    img1 = cv2.imread(image1_path,0)
    img2 = cv2.imread(image2_path,0)
    histMatch = BB.colors_histMatch(img1,img2)
    endtime = datetime.datetime.now()
    print(endtime - starttime)
    save_path = image2_path.replace('history.jpg', 'result1.jpg')
    cv2.imwrite('/space/code/multiview/v2/data/0407/fff.png', histMatch)
