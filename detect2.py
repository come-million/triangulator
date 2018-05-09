#!/usr/bin/env python

'''
Detect
===============================

detect detect detect 
detect detect detect 
detect detect detect 

Usage
-----
detect.py [<video_source>]


Keys
----
ESC   - exit
SPACE - start tracking
r     - toggle RANSAC
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np
import sys
import math

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


    # max_cos = np.max([angle_cos( c[i], c[(i+1) % 3], c[(i+2) % 3] ) for i in xrange(3)])                # 
    # if max_cos < 0.1:
    #   return True


if __name__ == '__main__':
    # print(__doc__)
    show=cv.imshow

    # try:
    #     fn = sys.argv[1]
    # except IndexError:
    fn1 = "in/shot_0_002.bmp"
    fn2 = "in/shot_0_007.bmp"
    
    # make dict {}
    rgb1=cv.imread(fn1)
    rgb2=cv.imread(fn2)
    yuv1=cv.cvtColor(rgb1, cv.COLOR_BGR2YUV)
    yuv2=cv.cvtColor(rgb2, cv.COLOR_BGR2YUV)
    hsv1=cv.cvtColor(rgb1, cv.COLOR_BGR2HSV)
    hsv2=cv.cvtColor(rgb2, cv.COLOR_BGR2HSV)

    (h1,s1,v1)=cv.split(hsv1)
    # show('h1',h1)
    # show('s1',s1)
    # show('v1',v1)

    (y1,u1,vv1)=cv.split(yuv1)
    # show('y1',y1)
    # show('u1',u1)
    # show('vv1',vv1)

    (h2,s2,v2)=cv.split(hsv2)
    # show('h2',h2)
    # show('s2',s2)
    # show('v2',v2)

    (y2,u2,vv2)=cv.split(yuv2)
    # show('y2',y2)
    # show('u2',u2)
    # show('vv2',vv2)

    # cv.waitKey(0)

    diff = cv.subtract(y1, y2)

    # TODO:
    # cv.erode()
    # cv.dilate()

    (mean, stddev) = cv.meanStdDev(diff)

    dst = cv.compare(diff - mean, 3 * stddev, cv.CMP_GT)
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    lines = cv.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)
    a,b,c = lines.shape
    for i in range(a):
        cv.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv.LINE_AA)

    # process loop
    def onChange(*arg):
        pass
    cv.namedWindow('edge')

    cv.createTrackbar('stddev_div_1000', 'edge', 4000, 6000, onChange)
    cv.createTrackbar('threshold', 'edge', 110, 200, onChange)
    cv.createTrackbar('minLineLength', 'edge', 60, 200, onChange)
    cv.createTrackbar('maxLineGap', 'edge', 20, 200, onChange)

    # cap = video.create_capture(fn)
    normdiff = diff - mean
    while True:
        # flag, img = cap.read()
        # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # edge = cv.Canny(src, thrs1, thrs2, apertureSize=7)
        # vis = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
        # vis = np.uint8(vis/2.)
        # vis[edge != 0] = (0, 255, 0)
        # cv.imshow('edge', vis)
        
        stddev_div_1000 = cv.getTrackbarPos('stddev_div_1000', 'edge')
        
        (mean, stddev) = cv.meanStdDev(diff)
        dst = cv.compare(normdiff, (stddev_div_1000 / 1000.0) * stddev, cv.CMP_GT)
        cdst = rgb1.copy()

        rho = 1
        theta = math.pi/180.0
        # threshold = 40
        threshold = cv.getTrackbarPos('threshold', 'edge')
        lines = np.array([])
        # minLineLength = 50
        minLineLength = cv.getTrackbarPos('minLineLength', 'edge')
        # maxLineGap = 10
        maxLineGap = cv.getTrackbarPos('maxLineGap', 'edge')

        lines = cv.HoughLinesP(dst, rho, theta, threshold, lines, minLineLength, maxLineGap)
        a,b,c = lines.shape
        for i in range(a):
            cv.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv.LINE_AA)
        
        cv.imshow('dst', dst)
        cv.imshow('edge', cdst)
        cv.moveWindow('edge', 99, 99)
        squares = []
        # bin=normdiff.copy()
        bin, contours, _hierarchy = cv.findContours(dst, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cnt_len = cv.arcLength(cnt, True)
            cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
            if cv.contourArea(cnt) > 1000:  # and cv.isContourConvex(cnt):
            # if len(cnt) == 3 and cv.contourArea(cnt) > 10 and cv.isContourConvex(cnt):
                # cnt = cnt.reshape(-1, 2)
                # max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 3], cnt[(i+2) % 3] ) for i in range(3)])
                # if max_cos < 0.1:
                squares.append(cnt)

        blobs = rgb1.copy()
        for i, c in enumerate(contours):
            cv.drawContours( blobs, contours, i, (i*120 % 255, i * 60 % 255, 127), -1 )

        cv.drawContours( blobs, squares, -1, (i*120 % 255, i * 60 % 255, 255), -1 )
        cv.imshow('blobs', blobs)
        cv.imwrite('blobs.png', blobs)

        ch = cv.waitKey(5)
        if ch == 27:
            break
    cv.destroyAllWindows()



# TODO:
# - get center of mass: findContours, squares.py
# - find corners: https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=cornerharris#goodfeaturestotrack
# - find angle: https://docs.opencv.org/2.4/modules/video/doc/motion_analysis_and_object_tracking.html#Mat%20estimateRigidTransform(InputArray%20src,%20InputArray%20dst,%20bool%20fullAffine)

# Utils 
# create trackbar (name, val, max, cb)

