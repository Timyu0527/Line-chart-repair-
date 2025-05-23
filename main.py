import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2.ximgproc as xip

X_D = 21
Y_D = 200 - 0
COLOR_DIFF = 19
THRESH_H = 5
THRESH_V = 15
KERNEL_SIZE = [3, 3, 3]
SAMPLE_NUM = '12'
SAMPLE_CLASS = '3'
SAMPLE_PATH = '.\\sample\\' + SAMPLE_NUM + '\\' + SAMPLE_NUM + '-' + SAMPLE_CLASS + '.png'
RESULT_TXT_PATH = '.\\result\\' + SAMPLE_NUM + '\\pixel_per_unit_' + SAMPLE_CLASS + '.txt'
RESULT_IMAGE_PATH= '.\\result\\' + SAMPLE_NUM + '\\' + SAMPLE_NUM + '-' + SAMPLE_CLASS + '.jpg'


# Read image
img = cv2.imread(SAMPLE_PATH)  # BGR format

IMAGE_H, IMAGE_W = img.shape[:2]

ori_img = img.copy()
ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
ori_img= cv2.bitwise_not(ori_img)

sub_img_h = np.array(ori_img)
_, sub_img_h = cv2.threshold(sub_img_h, THRESH_H, 255, cv2.THRESH_BINARY)
# # 圖中的摺線若與X軸相交可以使用
# sub_img_h = cv2.Canny(sub_img_h, threshold1=30, threshold2=100)
sub_img_h = xip.thinning(sub_img_h)


sub_img_v = np.array(ori_img[250:, :])
_, sub_img_v = cv2.threshold(sub_img_v, THRESH_V, 255, cv2.THRESH_BINARY)
sub_img_v = xip.thinning(sub_img_v)

plt.imshow(sub_img_v, cmap='gray')
plt.title("test")
plt.axis("off")
plt.show()

# lower = np.array([0, 201, 0])
# upper = np.array([255, 255, 255])

# # 建立遮罩
# img = cv2.inRange(img, lower, upper)
# img = cv2.bitwise_not(img)
# img[281:, :] = 0

# # 顯示結果
# plt.imshow(img, cmap='gray')
# plt.title('result')
# plt.axis('off')
# plt.show()

b, g, r = cv2.split(img)

# Caculate every pixel's max and min diff in RGB channel 
max_val = np.maximum(np.maximum(r, g), b)
min_val = np.minimum(np.minimum(r, g), b)
diff = max_val - min_val

# 建立遮罩：RGB 差異小於 diff
mask = diff < COLOR_DIFF

# 將這些像素設為白色
img[mask] = [255, 255, 255]


white_mask = np.all(img == [255, 255, 255], axis=-1)
img[~white_mask] = [0, 0, 0]

img = cv2.bitwise_not(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((KERNEL_SIZE[0], KERNEL_SIZE[0]), np.uint8))


kernel = np.ones((KERNEL_SIZE[1], KERNEL_SIZE[1]), np.uint8)
# img[:, :100] = cv2.dilate(img[:, :100], kernel, iterations=2)
# img[:, 170:] = cv2.dilate(img[:, 170:], kernel, iterations=3)
img = cv2.dilate(img, kernel, iterations=3)
# kernel = np.ones((KERNEL_SIZE[2], KERNEL_SIZE[2]), np.uint8)
# img = cv2.erode(img, kernel, iterations=2)
denoise_img = img.copy()

# skeletonize
skeleton = xip.thinning(img, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

# Use connected component to remove noise
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

min_area = 30
skeleton = np.zeros_like(skeleton)

for i in range(1, num_labels):  # 從 1 開始跳過背景
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        skeleton[labels == i] = 255

h_hough = np.zeros_like(sub_img_h)
v_hough = np.zeros_like(sub_img_v)

# Hough Line Transform
MAX_POINT_GAP = 0
h_lines = cv2.HoughLinesP(sub_img_h, 1, np.pi/180, threshold=30, minLineLength=20, maxLineGap=5)
horizontal_bound = []
for line in h_lines:
    x1, y1, x2, y2 = line[0]
    if abs(y1 - y2) <= MAX_POINT_GAP:
        horizontal_bound.append((x1, y1, x2, y2))
        cv2.line(h_hough, (x1, y1), (x2, y2), 255, 1)


v_lines = cv2.HoughLinesP(sub_img_v, 1, np.pi/180, threshold=2, minLineLength=5, maxLineGap=5)
vertical_bound = []

for line in v_lines:
    x1, y1, x2, y2 = line[0]
    if abs(x1 - x2) <= MAX_POINT_GAP:
        vertical_bound.append((x1, y1, x2, y2))
        cv2.line(v_hough, (x1, y1), (x2, y2), 255, 1)

# print(len(horizontal_bound))
# print(len(vertical_bound))
horizontal_bound= np.array(horizontal_bound)
h_y_values = np.array(horizontal_bound[:, 1])
print(horizontal_bound)
print(h_y_values)
h_y_max = np.max(h_y_values)
h_y_min = np.min(h_y_values)
origin_x = np.min(np.array(horizontal_bound[:, 0]))
origin_y = h_y_max
print(origin_x)
print(origin_y)
# print(h_y_min)
# print(h_y_max)

vertical_bound = np.array(vertical_bound)
v_x_values = np.array(vertical_bound[:, 0])
print(vertical_bound)
print(v_x_values)
v_x_max = np.max(v_x_values)
v_x_min = np.min(v_x_values)
print(v_x_min)
print(v_x_max)

kw_pre_pixel = (h_y_max - h_y_min) / Y_D
hr_pre_pixel = (v_x_max - v_x_min) / X_D

with open(RESULT_TXT_PATH, 'w', encoding='utf-8') as f:
    f.write(str(hr_pre_pixel) + '\n')
    f.write(str(kw_pre_pixel) + '\n')
    f.write(str(origin_x) + ' ' + str(origin_y))

cv2.imwrite(RESULT_IMAGE_PATH, skeleton)

# 儲存或顯示結果
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.imshow(h_hough, cmap='gray')
plt.title('Horizontal Hough')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(v_hough, cmap='gray')
plt.title('Vertical Hough')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(denoise_img, cmap='gray')
plt.title('Masked')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(skeleton, cmap='gray')
plt.title("Repaired Line")
plt.axis("off")
plt.show()