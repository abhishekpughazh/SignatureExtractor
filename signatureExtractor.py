from pdf2image import convert_from_path
import cv2
import matplotlib as plt
import numpy as np

def pdf_jpg(path):
  p = convert_from_path(path)
  p[0].save('image'+ '.jpg', 'JPEG')

pdf_jpg("F:/Downloads/testdocReal.pdf")

def sigExtract(inputPath, outputPath):
  # Load image and HSV color threshold
  image = cv2.imread(inputPath)
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  lower = np.array([10, 10, 0])
  upper = np.array([145, 500, 255])
  mask = cv2.inRange(hsv, lower, upper)
  result = cv2.bitwise_and(image, image, mask=mask)
  result[mask==0] = (255,255,255)

  # Find contours on extracted mask, combine boxes, and extract ROI
  cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  cnts = np.concatenate(cnts)
  x,y,w,h = cv2.boundingRect(cnts)
  cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
  # print(x, y, w, h)
  ROI = image[y:y+h, x:x+w]
  cv2.imwrite(outputPath, ROI)

  cv2.imshow('Result', result)
  cv2.imshow('Mask', mask)
  cv2.imshow('Image', image)
  cv2.imshow('Extracted Signature', ROI)
  cv2.waitKey(0)

sigExtract('image.jpg', 'final.png')