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

import cv2 as cv

if __name__ == '__main__':
    # print(__doc__)
    show=cv.imshow
    source='tileVideo.mp4'
    cap = cv.VideoCapture(source)

    while True: 
        flag, rgb1=cap.read()
        f = 0.25
        rgb1 = cv.resize(rgb1, (0,0),rgb1, f, f) 
        show('rgb1', rgb1)
        yuv1=cv.cvtColor(rgb1, cv.COLOR_BGR2YUV)
        show('yuv1', yuv1)
        hsv1=cv.cvtColor(rgb1, cv.COLOR_BGR2HSV)
        show('hsv1', hsv1)
        
        (h1,s1,v1)=cv.split(hsv1)
        show('h1',h1)
        show('s1',s1)
        show('v1',v1)

        (y1,u1,vv1)=cv.split(yuv1)
        show('y1',y1)
        show('u1',u1)
        show('vv1',vv1)

        ch = cv.waitKey(50)
        if ch == 27:
            # cv.imwrite('corners.png', corners)
            break
    cv.destroyAllWindows()