#!/usr/bin/python3

import cv2 as cv
import numpy as np


def chop(filename, p1, p2):
    cap = cv.VideoCapture(filename)

    buff = []
    score_hist = []
    while cap.isOpened():
        ret, frame = cap.read()

        roi = np.array(frame[p1[1]:p2[1], p1[0]:p2[0], :])

        roi_gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

        if len(buff) == 0:
            summe = np.zeros((roi_gray.shape), np.uint64)
            num_pixels = frame.shape[0] * frame.shape[1]

        summe += roi_gray
        buff.append(roi_gray)
        background = np.uint8(summe / float(len(buff)))

        if len(buff) > 10:
            summe -= buff.pop(0)

        diff = cv.absdiff(background, roi_gray)
        score = cv.norm(background, roi_gray, cv.NORM_L1) / float(num_pixels)

        score_hist.append(score)
        if len(score_hist) > 50:
            score_hist.pop(0)
        avg_score = np.average(score_hist)

        scaled = cv.resize(roi, (roi.shape[1]/10, roi.shape[0]/10))
                            interpolation=cv.INTER_NEAREST)

        cv.rectangle(frame, p1, p2, (0, 255, 0) if avg_score > 1.1 else (0, 0, 255), 3)

        Z = np.float32(scaled.reshape(-1, 3))
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        K = 20
        ret, label, center = cv.kmeans(Z, K, criteria, 5,
                                        cv.KMEANS_RANDOM_CENTERS)
        palette = np.reshape(np.uint8(center), (5, 4, 3))

        cv.imshow('palette', cv.resize(palette, (500, 400),
                   interpolation=cv.INTER_NEAREST))
        cv.imshow('original', frame)
        cv.imshow('roi', roi)
        cv.imshow('background', background)
        cv.imshow('diff', diff)

        if cv.waitKey(1) == 'q':
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    import sys

    print(cv.__version__)

    roi_dim = [int(i) for i in sys.argv[2].split(',')]
    chop(sys.argv[1], tuple(roi_dim[:2]), tuple(roi_dim[2:]))
