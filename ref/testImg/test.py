import cv2
import os

folder = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(folder, 'sc.png')

img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
if img is None:
	raise FileNotFoundError(f"Could not read image: {img_path}")

detector = cv2.ORB_create()
kps = detector.detect(gray)
print("Number of keypoints detected: ", len(kps))


out = cv2.drawKeypoints(
	img,
	kps,
	None,
	color=(0, 255, 0),
	flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)

# out_path = os.path.join(folder, 'sc_keypoints.png')
# cv2.imwrite(out_path, out)
# print("Saved keypoint image to:", out_path)

cv2.imshow('Image', img)
cv2.imshow("FAST keypoints", out)
cv2.waitKey(0)
cv2.destroyAllWindows()