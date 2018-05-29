#!/usr/bin/env python

# 743-840: green 'snake'

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


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def morph(img, ksize=(3, 3), iters=3):
    # cv.MORPH_ELLIPSE
    # cv.MORPH_RECT
    # cv.MORPH_CROSS
    st = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize)

    # cv.MORPH_ERODE
    # cv.MORPH_DILATE
    # cv.MORPH_OPEN
    # cv.MORPH_CLOSE
    # cv.MORPH_BLACKHAT
    # cv.MORPH_TOPHAT
    # cv.MORPH_GRADIENT
    res = cv.morphologyEx(img, cv.MORPH_ERODE, st, iterations=iters)
    return res

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

    source='data/tileVideo.mp4'
    cap = cv.VideoCapture(source)

    mser = cv.MSER_create()

    read_curr_index = -1
    read_next_index = 0
    processed_index = -1

    autoplay = False
    while True: 
        
        if autoplay:
            read_next_index += 1

        while read_curr_index != read_next_index:
            flag, rgb1=cap.read()
            # read_curr_index = read_next_index
            read_curr_index += 1
            


        if processed_index != read_curr_index:  # process: 
            processed_index = read_curr_index

            f = 0.5
            rgb1 = cv.resize(rgb1, (0,0), rgb1, f, f) 
            show('rgb1', rgb1)

            # gray = cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
            # yuv1=cv.cvtColor(rgb1, cv.COLOR_BGR2YUV)
            # show('yuv1', yuv1)
            hsv1=cv.cvtColor(rgb1, cv.COLOR_BGR2HSV)
            show('hsv1', hsv1)
            
            (h1,s1,v1)=cv.split(hsv1)
            show('h1',h1)
            show('s1',s1)
            show('v1',v1)

            # (y1,u1,vv1)=cv.split(yuv1)
            # show('y1',y1)
            # show('u1',u1)
            # show('vv1',vv1)

            # h2 = h1.copy()


            # gray = cv.cvtColor(rgb1, cv.COLOR_BGR2GRAY)
            vis = rgb1.copy()

            regions, _ = mser.detectRegions(h1)
            hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
            cv.polylines(vis, hulls, 1, (0, 255, 0))
            draw_str(vis, (10, 20), str(read_curr_index))
            show('mser', vis)

            ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 50)
            show('ada-25_50', ada)
            ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 40)
            show('ada-25_40', ada)
            ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 30)
            show('ada-25_30', ada)
            ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 20)
            show('ada-25_20', ada)

            # ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 51, -50)
            # show('ada-50_-50', ada)
            # ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 50)
            # show('ada-25-50', ada)
            # ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 0)
            # show('ada-25-0', ada)
            # ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 75, 0)
            # show('ada-75-0', ada)
            # ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 155, 0)
            # show('ada-155-0', ada)
            # ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 175, 0)
            # show('ada-175-0', ada)
            # ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 225, 0)
            # show('ada-225-0', ada)
            ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, -0)
            show('ada-25-0', ada)
            ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, -1)
            show('ada-25-1', ada)
            ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, -2)
            show('ada-25-2', ada)
            ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, -5)
            show('ada-25-5', ada)
            ada = cv.adaptiveThreshold(v1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 35, -20)
            show('ada-25-10', ada)



            (mean, stddev) = cv.meanStdDev(v1)
            dst0 = cv.compare(v1 - mean, 0 * stddev, cv.CMP_GT )
            dst1 = cv.compare(v1 - mean, 1 * stddev, cv.CMP_GT )
            dst2 = cv.compare(v1 - mean, 2 * stddev, cv.CMP_GT )
            dst3 = cv.compare(v1 - mean, 2.3 * stddev, cv.CMP_GT )
            show('dst0', dst0)
            show('dst1', dst1)
            show('dst2', dst2)
            show('dst3', dst3)
            cv.moveWindow('dst0', 0, 000)
            cv.moveWindow('dst1', 0, 100)
            cv.moveWindow('dst2', 0, 200)
            cv.moveWindow('dst3', 0, 300)


            morphed = morph(dst2)
            show('morphed', morphed)


            st = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
            res = cv.morphologyEx(dst3, cv.MORPH_OPEN, st, iterations=1)
            show('res', res)


            bin, contours, _hierarchy = cv.findContours(res, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            blobs = rgb1.copy()
            blobs[:]=0
            index_map=res.copy()
            index_map[:]=0

            triangles=[]
            for c in contours:
                if cv.contourArea(c) > 100:  # and cv.isContourConvex(cnt):
                    (center, radius) = cv.minEnclosingCircle(c)
                    x, y = int(center[0]), int(center[1])   
                    triangles.append({'x': x, 'y': y, 'r': radius, 'contour': c})

            # sort
            triangles.sort(key = lambda t: t['x'])
            tile_width = 5
            tile_height = 9
            for i in range(tile_width):
                a = i * tile_height
                b = (i+1) * tile_height
                triangles[a:b]=sorted(triangles[a:b], key = lambda t: t['y'])
                

            for i, t in enumerate(triangles):
                color = ((i*200) % 255, (i * 60) % 255, 127)
                cv.drawContours(blobs, t['contour'], 1, color, -1)
                cv.drawContours(index_map, t['contour'], 1, i, -1)

                x, y, radius = t['x'], t['y'], t['r']
                cv.circle(blobs, (x, y), int(radius), color, 5)
                cv.circle(blobs, (x, y), int(1), color, 5)
                draw_str(blobs, (x,y), "%d" % i)

            cv.imshow('blobs', blobs)
            cv.imshow('index_map', index_map)


            s=4
            patchsize = (s, s)
            
            # triangle center
            # TODO: get from tracker
            px = 600 * f
            py = 200 * f
            center = (px,py)
            zoom = cv.getRectSubPix(hsv1, patchsize, center)
            show('zoom', zoom)

            # t = 64
            t = sz//s
            # t*t = total patches

            i = read_curr_index
            ci=((i%(t*t))//t)*s
            cj=(i%t)*s
            # zoom[:,:,1] = 255  # s
            # zoom[:,:,2] = 255  # v

            rgb_zoom=cv.cvtColor(zoom, cv.COLOR_HSV2BGR)

            img[ci:ci+s, cj:cj+s, :] = rgb_zoom
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

        display_rate = 5  # frames
        # if not i % display_rate:
        ch = cv.waitKey(1)
        if ch == 27:
            # cv.imwrite('corners.png', corners)
            break
        
        if ch == ord('p'):
            autoplay = not autoplay

        if ch == ord('n'):
            read_next_index += 1

        if ch == ord('s'):
            cv.imwrite('screenshot.png', dst2)
            
    cv.destroyAllWindows()