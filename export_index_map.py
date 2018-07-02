#!/usr/bin/env python

import numpy as np
import cv2 as cv


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


if __name__ == '__main__':
    # print(__doc__)
    show=cv.imshow

    source='data/tileVideo.mp4'
    cap = cv.VideoCapture(source)

    # skip frames
    for i in range(690):
        cap.grab()

    flag, rgb1=cap.read()

    f = 1
    rgb1 = cv.resize(rgb1, (0,0), rgb1, f, f) 
    show('rgb1', rgb1)
    (r1,g1,b1)=cv.split(rgb1)
    cv.imwrite('r1.png',r1)
    cv.imwrite('g1.png',g1)
    cv.imwrite('b1.png',b1)

    hsv1=cv.cvtColor(rgb1, cv.COLOR_BGR2HSV)
    show('hsv1', hsv1)
    
    (h1,s1,v1)=cv.split(hsv1)
    show('h1',h1)
    show('s1',s1)
    show('v1',v1)
    cv.imwrite('h1.png', h1)
    cv.imwrite('s1.png', s1)
    cv.imwrite('v1.png', v1)



    yuv=cv.cvtColor(rgb1, cv.COLOR_BGR2YUV)
    (y,u,vv)=cv.split(yuv)
    show('y',y)
    show('u',u)
    show('vv',vv)
    cv.imwrite('y.png', y)
    cv.imwrite('u.png', u)
    cv.imwrite('vv.png', vv)


    
    t = v1
    (mean, stddev) = cv.meanStdDev(t)
    dst_low = cv.compare(t - mean, 0 * stddev, cv.CMP_GT )
    cv.imwrite('dst_low.png', dst_low)

    (mean, stddev) = cv.meanStdDev(t, mask=dst_low)

    dst3 = cv.compare(t - mean, 1 * stddev, cv.CMP_GT )
    show('dst3', dst3)
    cv.imwrite('dst3.png', dst3)


    # img = cv.medianBlur(y,5)
    # ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    # th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
    #             cv.THRESH_BINARY,11,2)
    th3 = cv.adaptiveThreshold(y,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,19,2)

    cv.imwrite('th3.png', th3)
    # st = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
    # res = cv.morphologyEx(dst3, cv.MORPH_ERODE, st, iterations=1)
    # res = dst3
    # res = th3 
    dst_and = cv.bitwise_and(dst3,dst_low)
    res = cv.bitwise_and(th3,dst_and)
    cv.imwrite('res.png', res)

    bin, contours, _hierarchy = cv.findContours(res, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    blobs = rgb1.copy()
    blobs[:]=0
    index_map=bin.copy()
    index_map[:]=0

    triangles=[]
    for i, c in enumerate(contours):
        if cv.contourArea(c) > 50:  # and cv.isContourConvex(cnt):
            (center, radius) = cv.minEnclosingCircle(c)
            x, y = int(center[0]), int(center[1])   
            triangles.append({'x': x, 'y': y, 'r': radius, 'i': i})

    # sort by x,y position  # removed once live registration (on-off pattern) is applied
    triangles.sort(key = lambda t: t['x'])
    tile_width = 5
    tile_height = 9
    for i in range(tile_width):
        a = i * tile_height
        b = (i+1) * tile_height
        triangles[a:b]=sorted(triangles[a:b], key = lambda t: t['y'])
        

    for i, t in enumerate(triangles):
        color = ((i*200) % 255, (i * 60) % 255, 127)
        cv.drawContours(blobs, contours, t['i'], color, cv.FILLED)
        cv.drawContours(index_map, contours, t['i'], int(i), cv.FILLED)

        x, y, radius = t['x'], t['y'], t['r']
        cv.circle(blobs, (x, y), int(radius), color, 5)
        cv.circle(blobs, (x, y), int(1), color, 5)
        cv.circle(index_map, (x, y), int(1), i, 5)
        draw_str(blobs, (x,y), "%d" % i)

    cv.imshow('blobs', blobs)
    cv.imshow('index_map', index_map)

    cv.imwrite('blobs.png', blobs)
    cv.imwrite('index_map.png', index_map)
            
    cv.imwrite('rgb.png', rgb1)

    cv.waitKey(0)

    cv.destroyAllWindows()