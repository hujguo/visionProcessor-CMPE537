import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
from sys import exit
import os
import random
from operator import sub

################################################################################
# Path to sets of images                                                       #
################################################################################

PREPATH = 'images/'
IMNAMES = {
    'cmpe_building': ['left_2', 'left_1', 'middle', 'right_1', 'right_2'],
    'north_campus': ['left_2', 'left_1', 'middle', 'right_1', 'right_2'],
    'paris': ['paris_a', 'paris_b', 'paris_c',],
    'book': ['book_table', 'book_orig'] # test dataset
}
EXT = '.jpg'

NPYDIR = 'data/'

RESDIR = 'results/'

METHODS = ['left-to-right', 'middle-out', 'first-out-then-middle']

################################################################################
# Save and load data                                                           #
################################################################################

def read_imgs(dataset):
    assert(IMNAMES.get(dataset))
    # read images
    imgs = []
    for name in IMNAMES[dataset]:
        path = PREPATH + dataset + '/' + name + EXT
        img = cv.imread(path, cv.IMREAD_COLOR)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        imgs.append(img)
    return imgs

# compress data
# no need to provide corresponding pairs for (j, i) where j > i
def save_points(dataset, points):
    N = len(IMNAMES.get(dataset))
    assert(N > 0)
    assert(len(points) == N)
    os.makedirs(NPYDIR, exist_ok=True)
    np.save(NPYDIR + dataset, np.array(points, dtype=object), allow_pickle=True)

# corresponding pointss for (j, i) where j > i are derived from (i, j) pair
def load_points(dataset):
    N = len(IMNAMES.get(dataset))
    assert(N > 0)
    points = np.load(NPYDIR + dataset + '.npy', allow_pickle=True)
    assert(len(points) == N)
    # fill in the blanks (we know A to B, obtain B to A)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            points[j][i] = [points[i][j][1], points[i][j][0]]
    return points

def save_image(name, img):
    os.makedirs(RESDIR, exist_ok=True)
    if cv.imwrite(RESDIR + name + EXT, cv.cvtColor(img, cv.COLOR_RGB2BGR)):
        print('Saved to', RESDIR + name + EXT)

def save_figure(fig, name):
    plt.figure(fig.number)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    fig.show()
    fig.savefig(RESDIR + name + '.png')
    mng.full_screen_toggle()
    print('Saved to ' + RESDIR + name + '.png')

################################################################################
# Corresponding points between images                                          #
################################################################################

def select_points(img, K):
    fig = plt.figure()
    plt.imshow(img)
    x = plt.ginput(K)   # (x, y) format
    plt.close(fig)
    return list(tuple(map(int, tup)) for tup in x)

def cv_mouse_click(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        pos = (x, y)    # cv2 format
        if type(param) is list:
            param.append(pos)

def cv_select_points(img, K, winname, param=[]):
    assert(type(param) is list)
    colors = color_list(K)
    cv.namedWindow(winname)
    cv.setMouseCallback(winname, cv_mouse_click, param)
    while True:
        try:
            cv.imshow(winname, mark_points(img, param, colors, bgr=True))
            ch = cv.waitKey(delay=100) & 0xFF
        # in case of an error, terminate the thread
        except Exception as e:
            print('Exception occured in', winname)
            print(e, flush=True)
            cv.destroyWindow(winname)
            exit(1)
        # press ESC to skip matching
        if ch is 27:
            if len(param) is not 0:
                print('Clear the points first!', flush=True)
            else:
                print('Skipping', winname, flush=True)
                break
        # press BACKSPACE to delete the last selected point
        elif ch is ord('\b'):
            if len(param):
                param.pop()
        # press SPACE when you are done
        elif ch is ord(' '):
            if len(param) is K:
                break
            else:
                print('There should be', K, 'selected points in', winname,
                      'whereas it is', len(param), flush=True)
    cv.destroyWindow(winname)
    return np.array(param)

def select_pair(img1, img2, K, win1, win2, pts1, pts2, print_help=True):
    if print_help:
        print('Select the corresponding %d points' %(K),
              'in each image in the same order')
        print('Press BACKSPACE to delete the last selected point')
        print('Press SPACE when you are done', flush=True)
    futures = [None, None]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures[0] = executor.submit(cv_select_points, img1, K, win1, pts1)
        futures[1] = executor.submit(cv_select_points, img2, K, win2, pts2)
    return [futures[0].result(), futures[1].result()]

def select_corresponding_points(dataset, K):
    assert(K >= 4)
    imgs = read_imgs(dataset)
    result = [[None for j in range(len(imgs))]
              for i in range(len(imgs))]
    print('Select the corresponding %d points' %(K),
          'in each image in the same order')
    print('Press ESC to skip matching that pair')
    print('Press BACKSPACE to delete the last selected point')
    print('Press SPACE when you are done', flush=True)
    for i in range(len(imgs) - 1):
        for j in range(i + 1, len(imgs)):
            win1 = IMNAMES[dataset][i]
            win2 = IMNAMES[dataset][j]
            img1 = cv.cvtColor(imgs[i], cv.COLOR_RGB2BGR)
            img2 = cv.cvtColor(imgs[j], cv.COLOR_RGB2BGR)
            print('Correspondence between', win1, 'and', win2, flush=True)
            result[i][j] = select_pair(img1, img2, K, win1, win2,
                                       [], [], print_help=False)
            # The following line is performed while loading data
            # result[j][i] = [result[i][j][1], result[i][j][0]]
    return result

def random_points(size, K):
    x = np.random.uniform(low=[0,0], high=[size[1], size[0]], size=(K, 2))
    return list(tuple(map(int, arr)) for arr in x)

################################################################################
# Homography calculation                                                       #
################################################################################

def computeH(src_pts, dst_pts, max_iter=1000, inlier_thr=5):
    assert(len(src_pts) >= 4)
    assert(len(dst_pts) == len(src_pts))
    # apply RANSAC algorithm
    best_inlier = 0     # number of points that are below threshold
    best_dist = float('inf')
    for _ in range(max_iter):
        # pick 4 random point pairs
        idx = random.sample(range(len(src_pts)), 4)
        # calculate homography matrix
        x1, x2, x3, x4 = ((src_pts[i][0], dst_pts[i][0]) for i in idx)
        y1, y2, y3, y4 = ((src_pts[i][1], dst_pts[i][1]) for i in idx)
        P = np.array([
            [-x1[0], -y1[0], -1, 0, 0, 0, x1[0] * x1[1], y1[0] * x1[1], x1[1]],
            [0, 0, 0, -x1[0], -y1[0], -1, x1[0] * y1[1], y1[0] * y1[1], y1[1]],
            [-x2[0], -y2[0], -1, 0, 0, 0, x2[0] * x2[1], y2[0] * x2[1], x2[1]],
            [0, 0, 0, -x2[0], -y2[0], -1, x2[0] * y2[1], y2[0] * y2[1], y2[1]],
            [-x3[0], -y3[0], -1, 0, 0, 0, x3[0] * x3[1], y3[0] * x3[1], x3[1]],
            [0, 0, 0, -x3[0], -y3[0], -1, x3[0] * y3[1], y3[0] * y3[1], y3[1]],
            [-x4[0], -y4[0], -1, 0, 0, 0, x4[0] * x4[1], y4[0] * x4[1], x4[1]],
            [0, 0, 0, -x4[0], -y4[0], -1, x4[0] * y4[1], y4[0] * y4[1], y4[1]],
            ])
        [U, S, Vt] = np.linalg.svd(P)
        H = Vt[-1].reshape(3, 3)
        H /= H[2][2]
        # see how good the matches are
        pts = transform(src_pts, H)
        distvec = np.sqrt(np.sum(np.square(pts - dst_pts), axis=1))
        dist = np.mean(distvec[distvec < inlier_thr])
        inlier = np.count_nonzero(distvec < inlier_thr)
        if inlier > best_inlier or (inlier is best_inlier and dist < best_dist):
            best_inlier = inlier
            best_dist = dist
            best_H = H
    return best_H

################################################################################
# Transformation functions                                                     #
################################################################################

def transform(src_pts, H):
    # src = [src_pts 1]
    src = np.pad(src_pts, [(0, 0), (0, 1)], constant_values=1)
    # pts = H * src
    pts = np.dot(H, src.T).T
    # normalize and throw z=1
    pts = (pts / pts[:,-1].reshape(-1, 1))[:, 0:2]
    return pts

# find the ROI of a transformation result
def warpRect(rect, H):
    x, y, w, h = rect
    corners = [[x, y], [x, y + h - 1], [x + w - 1, y], [x + w - 1, y + h - 1]]
    extremum = transform(corners, H)
    minx, miny = np.min(extremum[:,0]), np.min(extremum[:,1])
    maxx, maxy = np.max(extremum[:,0]), np.max(extremum[:,1])
    xo = int(np.floor(minx))
    yo = int(np.floor(miny))
    wo = int(np.ceil(maxx - minx))
    ho = int(np.ceil(maxy - miny))
    outrect = (xo, yo, wo, ho)
    return outrect

def size2rect(size):
    return (0, 0, size[1], size[0])

# homography matrix is translated to fit in the screen
def coverH(rect, H):
    # obtain bounding box of the result
    x, y, _, _ = warpRect(rect, H)
    # shift amount to the first quadrant
    xpos, ypos = int(0), int(0)
    if x < 0:
        xpos = int(-x)
    if y < 0:
        ypos = int(-y)
    # correct the homography matrix so that no point is thrown out
    T = np.array([[1, 0, xpos], [0, 1, ypos], [0, 0, 1]])
    H_corr = T.dot(H)
    return (H_corr, (xpos, ypos))

# pad image to cover ROI, return the shift amount of origin
def addBorder(img, rect):
    top, bottom, left, right = int(0), int(0), int(0), int(0)
    x, y, w, h = rect
    tl = (x, y)    
    br = (x + w, y + h)
    if tl[1] < 0:
        top = -tl[1]
    if br[1] > img.shape[0]:
        bottom = br[1] - img.shape[0]
    if tl[0] < 0:
        left = -tl[0]
    if br[0] > img.shape[1]:
        right = br[0] - img.shape[1]
    img = cv.copyMakeBorder(img, top, bottom, left, right,
                            cv.BORDER_CONSTANT, value=[0, 0, 0])
    orig = (left, top)
    return img, orig

def check_limits(pts, size):
    np.clip(pts[:,0], 0, size[1] - 1, pts[:,0])
    np.clip(pts[:,1], 0, size[0] - 1, pts[:,1])
    return pts

################################################################################
# Stitching functions                                                          #
################################################################################

def warpImage(img, H):
    # tweak the homography matrix to move the result to the first quadrant
    H_cover, pos = coverH(size2rect(img.shape), H)
    # find the bounding box of the output
    x, y, w, h = warpRect(size2rect(img.shape), H_cover)
    width, height = x + w, y + h
    assert(width * height < 1e8)    # do not exceed 300 MB for 8 GB RAM
    # warp the image using the corrected homography matrix
    # all the fuss is because of the indexing conventions of numpy and cv2
    # warped = cv.warpPerspective(img, H_corr, (width, height))
    idx_pts = np.mgrid[0:width, 0:height].reshape(2, -1).T
    map_pts = transform(idx_pts, np.linalg.inv(H_cover))
    map_pts = map_pts.reshape(width, height, 2).astype(np.float32)
    warped = cv.remap(img, map_pts, None, cv.INTER_CUBIC).transpose(1, 0, 2)
    # make the external boundary solid black, useful for masking
    warped = np.ascontiguousarray(warped, dtype=np.uint8)
    gray = cv.cvtColor(warped, cv.COLOR_RGB2GRAY)
    _, bw = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    # https://stackoverflow.com/a/55806272/12447766
    major = cv.__version__.split('.')[0]
    if major == '3':
        _, cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    else:
        cnts, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    warped = cv.drawContours(warped, cnts, 0, [0, 0, 0], lineType=cv.LINE_4)
    return (warped, pos)

# only the non-zero pixels are weighted to the average
def mean_blend(img1, img2):
    assert(img1.shape == img2.shape)
    locs1 = np.where(cv.cvtColor(img1, cv.COLOR_RGB2GRAY) != 0)
    blended1 = np.copy(img2)
    blended1[locs1[0], locs1[1]] = img1[locs1[0], locs1[1]]
    locs2 = np.where(cv.cvtColor(img2, cv.COLOR_RGB2GRAY) != 0)
    blended2 = np.copy(img1)
    blended2[locs2[0], locs2[1]] = img2[locs2[0], locs2[1]]
    blended = cv.addWeighted(blended1, 0.5, blended2, 0.5, 0)
    return blended

def blend_images(imageA, imageB, H):
    return warpPano(imageA, imageB, H, (0, 0))

def cv_blend_images(imageA, imageB, H):
    # move origin to cover the third quadrant
    H_corr, pos = coverH(size2rect(imageA.shape), H)
    xpos, ypos = pos
    # warp the image and paste the original one
    result = cv.warpPerspective(imageA, H_corr, (5000, 5000))
    bottom, right = int(0), int(0)
    if ypos + imageB.shape[0] > result.shape[0]:
        bottom = ypos + imageB.shape[0] - result.shape[0]
    if xpos + imageB.shape[1] > result.shape[1]:
        right = xpos + imageB.shape[1] - result.shape[1]
    result = cv.copyMakeBorder(result, 0, bottom, 0, right,
                               cv.BORDER_CONSTANT, value=[0, 0, 0])
    # mean value blending
    idx = np.s_[ypos:ypos+imageB.shape[0], xpos:xpos+imageB.shape[1]]
    result[idx] = mean_blend(result[idx], imageB)
    # crop extra paddings
    x,y,w,h = cv.boundingRect(cv.cvtColor(result, cv.COLOR_RGB2GRAY))
    result = result[0:y+h,0:x+w]
    # return the resulting image with shift amount
    return (result, (xpos, ypos))

def warpPano(prevPano, img, H, orig):
    # corret homography matrix
    T = np.array([[1, 0, -orig[0]], [0, 1, -orig[1]], [0, 0, 1]])
    H_corr = H.dot(T)
    # warp the image and obtain shift amount of origin
    result, pos = warpImage(prevPano, H_corr)
    xpos, ypos = pos
    # zero pad the result
    rect = (xpos, ypos, img.shape[1], img.shape[0])
    result, _ = addBorder(result, rect)
    # mean value blending
    idx = np.s_[ypos : ypos + img.shape[0], xpos : xpos + img.shape[1]]
    result[idx] = mean_blend(result[idx], img)
    # crop extra paddings
    x, y, w, h = cv.boundingRect(cv.cvtColor(result, cv.COLOR_RGB2GRAY))
    result = result[y : y + h, x : x + w]
    # return the resulting image with shift amount
    return (result, (xpos - x, ypos - y))

# no warping here, useful for combining two different stitched images
# the image at given origin coordinates must be the same
def patchPano(img1, img2, orig1=(0,0), orig2=(0,0)):
    # bottom right points
    br1 = (img1.shape[1] - 1, img1.shape[0] - 1)
    br2 = (img2.shape[1] - 1, img2.shape[0] - 1)
    # distance from orig to br
    diag2 = tuple(map(sub, br2, orig2))
    # possible pano corner coordinates based on img1
    extremum = np.array([(0, 0), br1,
                tuple(map(sum, zip(orig1, diag2))),
                tuple(map(sub, orig1, orig2))])
    bb = cv.boundingRect(extremum)
    # patch img1 to img2
    pano, shift = addBorder(img1, bb)
    orig = tuple