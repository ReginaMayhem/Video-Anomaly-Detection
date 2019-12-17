import os
import cv2 as cv
import numpy as np
import flow_vis


def getcell(matrix, starti, startj):
    ret = []
    for i in range(4 * starti, 4 * starti + 10):
        tmp = []
        for j in range(4 * startj, 4 * startj + 10):
            tmp.append(matrix[i][j])
        ret.append(tmp)
    return ret


def HOF(magnitude, angle):
    ret = []
    unitdegree = np.pi / 4
    for i in range(int((len(magnitude) - 10) / 4) + 1):
        for j in range(int((len(magnitude[0]) - 10) / 4) + 1):
            cellmagnitude = getcell(magnitude, i, j)
            cellangle = getcell(angle, i, j)
            mydict = {new_list: [] for new_list in range(8)}
            for subi in range(len(cellmagnitude)):
                for subj in range(len(cellmagnitude[0])):
                    num_rad = int(cellangle[subi][subj] // unitdegree)
                    print(subi, subj)
                    print('len(cellmagnitude)', len(cellmagnitude))
                    print('len(cellmagnitude[0])', len(cellmagnitude[0]))
                    mydict[num_rad].append(cellmagnitude[subi][subj])
            assert(len(mydict[1]) + len(mydict[2]) + len(mydict[3]) + len(mydict[4]) + len(mydict[5]) + len(mydict[6]) + len(mydict[7]) + len(mydict[0]) == 100)
            ret.append(mydict)
    return ret


def normalize(mylist):
    maxv = [0] * 8
    for mydict in mylist:
        for i in range(len(maxv)):
            maxv[i] = max(maxv[i], max(mydict[i]))
    return mylist


def to_hist(mylist):
    mylist = normalize(mylist)
    ret = []
    for mydict in mylist:
        tmpdict = {new_list: [0] * 10 for new_list in range(8)}
        for i in range(len(mydict)):
            for j in range(mydict[i]):
                index = mydict[i][j] / 0.1
                tmpdict[i][index] += 1
        ret.append(tmpdict)
    return ret


cwd = os.getcwd()
subfolders = [dir for dir in os.listdir() if dir[:5] == "Train"]
mylist = []
for subfolder in sorted(subfolders):
    path = cwd + '/' + subfolder
    prev = None
    for img in sorted(os.listdir(path)):
        if img[-3:] != 'tif':
            continue
        frame = cv.imread(path + '/' + img, 0)
        if prev is not None:
            flow = cv.calcOpticalFlowFarneback(prev, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])  # return angle in radians
            mylist = mylist + HOF(magnitude, angle)
            # flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
            # cv.imshow('flow', flow_color)
            # cv.waitKey(1)
        # cv.imshow('image', frame)
        # cv.waitKey(1)

        prev = frame

print(len(mylist))
