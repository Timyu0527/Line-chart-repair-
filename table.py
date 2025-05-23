import cv2
import numpy as np
import matplotlib.pyplot as plt

THRESH = 85
CLASS_COLOR = np.array([
    [240, 203, 155],
    [155, 155, 245],
    [130, 190, 250],
    [20, 230, 250],
    [135, 205, 140],
    [205, 165, 205]
])
CLASS_NUM = 6
ESP = 13
MAX_LINE_GAP = 10
MAX_POINT_GAP = 0

for num in range(1, 15):
    count = [0, 0, 0, 0, 0, 0]
    SAMPLE_NUM = str(num)
    SAMPLE_PATH = '.\\sample\\' + SAMPLE_NUM + '\\' + SAMPLE_NUM + '_out.png'

    img = cv2.imread(SAMPLE_PATH)

    IMAGE_H, IMAGE_W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    gray = cv2.Canny(gray, 100, 100)
    _, gray = cv2.threshold(gray, THRESH, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(gray, 1, np.pi/180, threshold=30, minLineLength=100, maxLineGap=20)

    hough = np.zeros_like(gray)
    vertical = []
    horizontal = []


    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) <= MAX_POINT_GAP:
            vertical.append(x1)
            # cv2.line(hough, (x1, y1), (x2, y2), 255, 1)
        if abs(y1 - y2) <= MAX_POINT_GAP:
            horizontal.append(y1)
            # cv2.line(hough, (x1, y1), (x2, y2), 255, 1)

    vertical = np.array(sorted(vertical))
    vertical_selected = [vertical[0]]

    cv2.line(hough, (vertical[0], 0), (vertical[0], IMAGE_H), 255, 1)
    for i in range(1, len(vertical)):
        if vertical[i] - vertical[i - 1] >= MAX_LINE_GAP:
            vertical_selected.append(vertical[i])
            cv2.line(hough, (vertical[i], 0), (vertical[i], IMAGE_H), 255, 1)

    print(len(vertical_selected))

    horizontal = np.array(sorted(horizontal))
    horizontal_selected = [horizontal[0]]

    cv2.line(hough, (0, horizontal[0]), (IMAGE_W, horizontal[0]), 255, 1)
    for i in range(1, len(horizontal)):
        if horizontal[i] - horizontal[i - 1] >= MAX_LINE_GAP:
            horizontal_selected.append(horizontal[i])
            cv2.line(hough, (0, horizontal[i]), (IMAGE_W, horizontal[i]), 255, 1)

    print(len(horizontal_selected))

    result = np.zeros_like(img)

    for i in range(1, len(vertical_selected)):
        x = (vertical_selected[i] + vertical_selected[i - 1]) // 2
        for j in range(1, len(horizontal_selected)):
            y = (horizontal_selected[j] + horizontal_selected[j - 1]) // 2
            pixel = tuple(int(i) for i in img[y, x])
            for k in range(CLASS_NUM):
                if all(CLASS_COLOR[k] - ESP <= pixel) and all(pixel <= CLASS_COLOR[k] + ESP):
                    pixel = pixel[::-1]
                    # print(pixel)
                    # print(x, y)
                    cv2.circle(result, (x, y), 15, pixel, -1)
                    count[k] += 1

    ans = []

    with open('table.ans', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            ans.append([int(x) for x in line.split()])

    for i in ans:
        sum = np.sum(i)
        if sum != 365:
            raise ValueError('Sum of a year was worng.')

    print(count)
    print(ans[int(SAMPLE_NUM) - 1])
    if count != ans[int(SAMPLE_NUM) - 1]:
        raise ValueError(str(num) + 'th ans was worng.')

    plt.figure(figsize=(8, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(hough, cmap='gray')
    plt.title("hough")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.title("result")
    plt.axis("off")
    plt.show()