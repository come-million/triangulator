import cv2 as cv
import numpy as np

if __name__ == '__main__':
    def onChange(*arg):
        pass
    cv.namedWindow('edge')
    cv.moveWindow('edge', 99, 99)
    
    cv.createTrackbar('t1', 'edge', 10, 6000, onChange)
    cv.createTrackbar('t2', 'edge', 50, 6000, onChange)

    source = 0
    cap = cv.VideoCapture(source)
    # if 'size' in params:
        # w, h = map(int, params['size'].split('x'))
        # cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)
    if cap and cap.isOpened():
        print('Warning: unable to open video source: ', source)
        # if fallback is not None:
        #     #TODO: retry 
        # ### return create_capture(fallback, None)

    while True:
        flag, img = cap.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        t1 = cv.getTrackbarPos('t1', 'edge')
        t2 = cv.getTrackbarPos('t2', 'edge')
        edge = cv.Canny(gray, t1, t2, apertureSize=7)
        cv.imshow('edge', edge)
        
        vis = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        cv.imshow('vis', vis)
        
        ch = cv.waitKey(5)
        if ch == 27:
            cv.imwrite('edge.png', edge)
            break
    cv.destroyAllWindows()
