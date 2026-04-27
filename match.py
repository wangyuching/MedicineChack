import os
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(current_dir, 'testPic', 'bothClose.png')
img = cv2.imread(img_path)
img2 = img.copy()

template_path = os.path.join(current_dir, 'testPic', 'wordBF.png')
template = cv2.imread(template_path)
template.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
thereshold = 0.8
loc = np.where(res >= thereshold)

rects = []
for pt in zip(*loc[::-1]):
    rects.append([pt[0], pt[1], w, h])

rests, weights = cv2.groupRectangles(rects, groupThreshold = 1, eps = 0.5)

for (x, y, w, h) in rects:
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('Result', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()





# methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR', 'TM_CCORR_NORMED', 
#            'TM_SQDIFF', 'TM_SQDIFF_NORMED']

# for meth in methods:
#     img = img2.copy()
#     method = getattr(cv2, meth)

#     res = cv2.matchTemplate(img, template, method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#         bottom_right = (top_left[0] + w, top_left[1] + h)

#         cv2.rectangle(img, top_left, bottom_right, 255, 2)

# plt.subplot(121),plt.imshow(res,cmap = 'gray')
# plt.title('匹配結果'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(img,cmap = 'gray')
# plt.title('檢測點'), plt.xticks([]), plt.yticks([])
# plt.suptitle(meth)
 
# plt.show()
