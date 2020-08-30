#!/usr/bin/python3

import cv2 as cv
import numpy as np


def chop(filename, p1, p2):
    cap = cv.VideoCapture(filename)

    buff_gray = []
    buff_saturation = []
    score_hist = []
    while cap.isOpened():
        ret, frame = cap.read()

        roi = np.array(frame[p1[1]:p2[1], p1[0]:p2[0], :])

        roi_saturation = cv.cvtColor(roi, cv.COLOR_BGR2HSV)[:,:,1]
        roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

        if len(buff_gray) == 0:
            summe_gray = np.zeros((roi_gray.shape), np.uint64)
            summe_saturation = np.zeros((roi_gray.shape), np.uint64)
            num_pixels = frame.shape[0] * frame.shape[1]

        summe_gray += roi_gray
        summe_saturation += roi_saturation
        buff_gray.append(roi_gray)
        buff_saturation.append(roi_saturation)
        background = np.uint8(summe_gray / float(len(buff_gray)))
        background_saturation = np.uint8(summe_gray / float(len(buff_saturation)))

        if len(buff_gray) > 10:
            summe_gray -= buff_gray.pop(0)

        diff = cv.absdiff(background, roi_gray)
        saturation_diff = cv.absdiff(background_saturation, roi_saturation)
        diff_product = np.uint8(255 * (diff / 255 * saturation_diff / 255))
        score = cv.norm(diff_product, cv.NORM_L1) / float(num_pixels)
        print(score)

        score_hist.append(score)
        if len(score_hist) > 50:
            score_hist.pop(0)
        avg_score = np.average(score_hist)

        scaled = cv.resize(roi, (int(roi.shape[1]/10), int(roi.shape[0]/10)),
                            interpolation=cv.INTER_NEAREST)

        cv.rectangle(frame, p1, p2, (0, 255, 0) if avg_score > 0.03 else (0, 0, 255), 3)

        Z = np.float32(scaled.reshape(-1, 3))
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        K = 20
        ret, label, center = cv.kmeans(Z, K, None, criteria, 5,
                                       cv.KMEANS_RANDOM_CENTERS)
        palette = np.reshape(np.uint8(center), (5, 4, 3))

        cv.imshow('palette', cv.resize(palette, (500, 400),
                   interpolation=cv.INTER_NEAREST))

        cv.imshow('original', cv.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2))))
        cv.imshow('roi', roi)
        cv.imshow('background', background)
        cv.imshow('diff_luminance', diff)
        cv.imshow('diff_saturation', saturation_diff)
        cv.imshow('diff_product', diff_product)

        if cv.waitKey(1) == 'q':
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    import sys

    print(cv.__version__)

    roi_dim = [int(i) for i in sys.argv[2].split(',')]
    chop(sys.argv[1], tuple(roi_dim[:2]), tuple(roi_dim[2:]))
