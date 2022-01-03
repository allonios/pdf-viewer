import cv2

def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp, window_name = "window"):
    return cv2.imshow(window_name,cv2.drawKeypoints(gray_img, kp, color_img.copy()))

img = cv2.imread('./EyeDatabase/eye13.jpg')
img2 = cv2.imread('./EyeDatabase/eye0.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

kp, desc = gen_sift_features(gray)
kp2, desc2 = gen_sift_features(gray)
show_sift_features(gray, img, kp, "window1");
show_sift_features(gray2, img2, kp2, "window2");



bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

matches = bf.match(desc, desc2)

# Sort the matches in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# draw the top N matches
N_MATCHES = 100

match_img = cv2.drawMatches(
    img, kp,
    img2, kp2,
    matches[:N_MATCHES], img2.copy(), flags=0)

cv2.imshow("matches",match_img);



cv2.waitKey(0)





