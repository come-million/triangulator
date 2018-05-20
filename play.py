#!/usr/bin/env python

'''
play
===============================

play play play 
play play play 
play play play 

Usage
-----
play.py [<video_source>]


Keys
----
ESC   - exit
SPACE - start something
r     - toggle
'''

import numpy as np
import cv2 as cv


# small = cv.pyrDown(small)

if __name__ == '__main__':
    # print(__doc__)
    show=cv.imshow

    sz = 1024
    img = np.zeros((sz, sz, 3), np.uint8)
    track = np.cumsum(np.random.rand(500000, 2)-0.5, axis=0)
    track = np.int32(track*100 + (sz/2, sz/2))
    cv.polylines(img, [track], 0, 255, 1, cv.LINE_AA)
    # show('img',img)
    # cv.waitKey(0)

    source='tileVideo.mp4'
    cap = cv.VideoCapture(source)
    i = -1
    while True: 
        i += 1
        flag, rgb1=cap.read()
        f = 0.25
        rgb1 = cv.resize(rgb1, (0,0), rgb1, f, f) 
        show('rgb1', rgb1)
        yuv1=cv.cvtColor(rgb1, cv.COLOR_BGR2YUV)
        show('yuv1', yuv1)
        hsv1=cv.cvtColor(rgb1, cv.COLOR_BGR2HSV)
        show('hsv1', hsv1)
        
        # (h1,s1,v1)=cv.split(hsv1)
        # show('h1',h1)
        # show('s1',s1)
        # show('v1',v1)

        # (y1,u1,vv1)=cv.split(yuv1)
        # show('y1',y1)
        # show('u1',u1)
        # show('vv1',vv1)

        # h2 = h1.copy()


        s=32
        patchsize = (s, s)
        
        px = 150
        py = 50
        center = (px,py)
        zoom = cv.getRectSubPix(hsv1, patchsize, center)
        show('zoom', zoom)

        cx=((i%sz)//s)*s
        cy=(i%s)*s
        zoom[:,:,1] = 255  # s
        zoom[:,:,2] = 255  # v

        rgb_zoom=cv.cvtColor(zoom, cv.COLOR_HSV2BGR)

        img[cx:cx+32, cy:cy+32, :] = rgb_zoom
        show('img',img)


        # mean, stdev = cv.meanStdDev(zoom)
        # ps = (patchsize / 2.0, patchsize / 2.0)
        # cv.rectangle(h1, center - ps, center + ps, 222, 1)
        # show('h1b',h1)
        # print(mean)
        # hsv_mat = np.zeros((1, 1, 3), np.uint8)
        # pix = mean[0][0]
        # pix = mean[0:2][0]

        # print(hsv_mat)
        # mean_to_rgb=cv.cvtColor(hsv_mat, cv.COLOR_HSV2BGR)
        # print(mean_to_rgb)
        
        # c=(mean_to_rgb[0][0][0], mean_to_rgb[0][0][1], mean_to_rgb[0][0][2])

        # cv.rectangle(img, (100, 100+i), (200,200+i), pix, 1)
        ch = cv.waitKey(10)
        if ch == 27:
            # cv.imwrite('corners.png', corners)
            break
    cv.destroyAllWindows()