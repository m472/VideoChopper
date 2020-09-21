#!/usr/bin/python3

import cv2 as cv
import numpy as np

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)


def chop(filename, p1, p2):
    cap = cv.VideoCapture(filename)

    score_hist = []
    while cap.isOpened():
        ret, frame = cap.read()

        roi_rgb = np.array(frame[p1[1]:p2[1], p1[0]:p2[0], :])

        scaled = cv.resize(roi_rgb, (int(roi_rgb.shape[1]/10), int(roi_rgb.shape[0]/10)),
                           interpolation=cv.INTER_NEAREST)

        Z = np.float32(scaled.reshape(-1, 3))
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 5, 1.0)

        palette_shape = (5, 4, 3)
        K = palette_shape[0] * palette_shape[1]
        ret, label, center = cv.kmeans(Z, K, None, criteria, 5,
                                       cv.KMEANS_RANDOM_CENTERS)
        palette = np.uint8(np.sort(center, axis=2))

        score = np.average(cv.cvtColor(np.reshape(palette, palette_shape), cv.COLOR_BGR2HSV)[:, :, 1])
        score_hist.append(score)
        if len(score_hist) > 50:
            score_hist.pop(0)
        avg_score = np.average(score_hist)

        print(avg_score)
        cv.rectangle(frame, p1, p2, COLOR_GREEN if avg_score > 20 else COLOR_RED, 3)

        cv.imshow('palette', cv.resize(np.reshape(palette, palette_shape), (500, 400),
                                       interpolation=cv.INTER_NEAREST))

        cv.imshow('original', cv.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2))))
        cv.imshow('roi', roi_rgb)

        if cv.waitKey(1) == 'q':
            break

    cv.destroyAllWindows()


if __name__ == "__main__":
    import sys

    roi_dim = [int(i) for i in sys.argv[2].split(',')]
    chop(sys.argv[1], tuple(roi_dim[:2]), tuple(roi_dim[2:]))
