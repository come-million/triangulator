import cv2 as cv

if __name__ == '__main__':

    a=cv.imread("data/shot_0_000.bmp")
    cv.imwrite("out/shot_0_000.jpg", a)
    cv.imwrite("out/shot_0_000.png", a)
    
    x=(cv.IMWRITE_PNG_COMPRESSION, 9)
    cv.imwrite("out/shot_0_000-c9.png", a, x)
    cv.imwrite("out/shot_0_000-c9.jp2", a)

    cv.imshow("a", a)
    cv.waitKey(10)
    
