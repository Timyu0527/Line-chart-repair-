# Color Mask
透過顏色遮罩選出目標折線，主要有兩種方法，擇一使用。後續再使用骨架化及去雜訊形成最終圖像。
## Method 1
使用BGR色域，將目標折線遮蓋，並反轉圖片顏色，藍色、紫色或綠色比較需要用Method 1。底下lower非0部分建議200~210效果較佳，值越小雜訊越小。
```python
# blue
lower = np.array([0, 0, 202])
upper = np.array([255, 255, 255])
# purple
lower = np.array([0, 201, 0])
upper = np.array([255, 255, 255])

# 建立遮罩
img = cv2.inRange(img, lower, upper)
img = cv2.bitwise_not(img)
```
## Method 2
計算所有pixel的BGR最大差值，大於COLOR_DIFF則為目標折線。此方法對於資料集中大部分顏色效果都不錯，但在藍色、紫色和綠色時，有時效果較差。
```python
b, g, r = cv2.split(img)

# 計算所有pixel的最大差值
max_val = np.maximum(np.maximum(r, g), b)
min_val = np.minimum(np.minimum(r, g), b)
diff = max_val - min_val

# 建立遮罩：BGR 差異小於 COLOR_DIFF
mask = diff < COLOR_DIFF

# 將這些像素設為白色
img[mask] = [255, 255, 255]


white_mask = np.all(img == [255, 255, 255], axis=-1)
img[~white_mask] = [0, 0, 0]

img = cv2.bitwise_not(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```
## 參數

- **THRESH_H, THRESH_V**：值越大雜訊越小，建議值 `[0, 5, 10, 15]`
- **COLOR_DIFF(for Method 2)**：值越大雜訊越小

|Color|light blue|red|orange|yellow|purple|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Value|1~20|1~20|1~20 or 95~100|1~25 or 140 up|1~10|

- **iteration**：建議1~3

- **KERNEL_SIZE**：建議3、5
