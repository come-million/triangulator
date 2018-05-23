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

if __name__ == '__main__':
    # print(__doc__)
    show=cv.imshow

    # try:
    #     fn = sys.argv[1]
    # except IndexError:
    fn1 = "data/shot_0_002.bmp"
    fn2 = "data/shot_0_007.bmp"
    
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

    # import pdb;pdb.set_trace()
    # rgb = cv.cvtColor(yuv2, cv.COLOR_YUV2BGR)
    # gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)

    src = diff
    dst = cv.compare(diff - mean, 3 * stddev, cv.CMP_GT )

    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    lines = cv.HoughLinesP(dst, 1, math.pi/180.0, 40, np.array([]), 50, 10)
    a,b,c = lines.shape
    for i in range(a):
        cv.line(cdst, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("detected", cdst)

    cv.imshow("source", src)

    cv.imwrite("out/src.png", src)
    cv.imwrite("out/dst.png", dst)
    cv.imwrite("out/cdst.png", cdst)

    cv.waitKey(0)


    # process loop
    def nothing(*arg):
        pass
    cv.namedWindow('edge')

    cv.createTrackbar('stddev_div_1000', 'edge', 4000, 6000, nothing)
    cv.createTrackbar('threshold', 'edge', 110, 200, nothing)
    cv.createTrackbar('minLineLength', 'edge', 60, 200, nothing)
    cv.createTrackbar('maxLineGap', 'edge', 20, 200, nothing)

    # cap = video.create_capture(fn)
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
        dst = cv.compare(diff - mean, (stddev_div_1000 / 1000.0) * stddev, cv.CMP_GT)
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

        ch = cv.waitKey(5)
        if ch == 27:
            break
    cv.destroyAllWindows()

