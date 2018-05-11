# opencv-live.py
# a quick and easy

import cv2 as cv


def make_window(title):
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.moveWindow(title, x=199, y=99)

def onChange(*arg):
    pass

class Canny():

    def __init__(self, name):
        self.name = name

    def setup():
        cv.createTrackbar('t1', self.name, 10, 200, onChange)
        cv.createTrackbar('t2', self.name, 50, 500, onChange)

    def do(gray):
        t1 = cv.getTrackbarPos('t1', 'edge')
        t2 = cv.getTrackbarPos('t2', 'edge')
        self.edge = cv.Canny(gray, t1, t2, apertureSize=7)

    def show():
        cv.imshow(self.name, edge)

def loop_capture(window, source):

    cap = cv.VideoCapture(source)
    if cap and cap.isOpened():
        print('Opened: ', source)
    else:
        print('Warning: unable to open video source: ', source)
        raise


    canny = Canny('edges')

    
    while True:
    
        flag, img = cap.read()
        cv.imshow(title, img)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        canny.do(gray)
        canny.show()
        
        vis = img.copy()
        cv.imshow('vis', vis)
        
        ch = cv.waitKey(1)
        if ch == 27:
            cv.imwrite(title + '.png', edge)
            # add png compression?
        break

    

if __name__ == '__main__':

    window=make_window(title='live')
    loop_capture(source=0, window=window)
    cv.destroyAllWindows()
