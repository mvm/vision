#!/usr/bin/env python3

import sys
from math import inf
from time import perf_counter

import cv2 as cv
import numpy as np

if len(sys.argv) != 2:
    print("Usage: %s [file]" % (sys.argv[0]))
    sys.exit(-1)

img_name = sys.argv[1]
img = cv.imread(img_name)

if img is None:
    print("Error abriendo la imagen '%s'" % (img_name))
    sys.exit(-1)

img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# convert image to binary
cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY, img_gray)

img = img_gray

def meijster_iter_col(img):
    out = np.zeros_like(img_gray, dtype=np.float32)
    for col in range(0, img.shape[1]):
        out[:, col] = img[:, col]
        b = 1
        for row in range(1, img.shape[0]):
            if img[row, col] != 0:
                out[row, col] = out[row-1, col] + b
                b += 2
            else:
                out[row, col] = 0
                b = 1
        b = 1
        for row in reversed(range(0, img.shape[0]-1)):
            if out[row, col] > out[row+1, col]:
                out[row, col] = out[row+1, col] + b
                b += 2
            if out[row,col] == 0:
                b = 1
    return out

def felzen_row(row):
    out = np.zeros_like(row)
    k = 0
    v = np.zeros(row.shape[0] + 1, dtype=np.uint32)
    z = np.zeros(row.shape[0] + 1, dtype=np.float32)
    v[0] = 0
    z[0] = -inf
    z[1] = inf
    i = 1
    while i < row.shape[0]:
        while (s := ((row[i] + i**2) - (row[v[k]] + v[k]**2)) / (2*i - 2*v[k])) <= z[k]:
            k = k - 1
        if s > z[k]:
            k = k + 1
            v[k] = i
            z[k] = s
            z[k+1] = inf
        i = i + 1
    k = 0
    for i in range(0, row.shape[0]):
        while z[k+1] < i:
            k = k + 1
        out[i] = (i - v[k])**2 + row[v[k]]
    return out

t_before = perf_counter()
out = meijster_iter_col(img)

for j in range(0, out.shape[0]):
    out[j] = felzen_row(out[j])
t_after = perf_counter()
print("Time: %f s" % (t_after - t_before))

out = np.sqrt(out)
max_color = np.max(out)
out = out * 255.0 / max_color

cv.imwrite("edt.png", out)
