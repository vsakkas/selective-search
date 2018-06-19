import numpy as np
from skimage import color, feature, segmentation, util


def _generate_segments(im_orig: np.ndarray, scale: float, sigma: float, min_size: int) -> np.ndarray:
    '''
    Segment smallest regions by the algorithm of Felzenswalb and Huttenlocher
    '''
    # open the Image
    im_mask = segmentation.felzenszwalb(
        util.img_as_float(im_orig), scale=scale, sigma=sigma,
        min_size=min_size)

    # merge mask channel to the image as a 4th channel
    im_orig = np.append(
        im_orig, np.zeros(im_orig.shape[:2])[:, :, np.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask

    return im_orig


def _sim_color(r1: dict, r2: dict) -> np.float64:
    '''
    Calculate the sum of histogram intersection of color
    '''
    return sum([min(a, b) for a, b in zip(r1['hist_c'], r2['hist_c'])])


def _sim_texture(r1: dict, r2: dict) -> np.float64:
    '''
    Calculate the sum of histogram intersection of texture
    '''
    return sum([min(a, b) for a, b in zip(r1['hist_t'], r2['hist_t'])])


def _sim_size(r1: dict, r2: dict, imsize: int) -> float:
    '''
    Calculate the size similarity over the image
    '''
    return 1.0 - (r1['size'] + r2['size']) / imsize


def _sim_fill(r1: dict, r2: dict, imsize: int) -> float:
    '''
    Calculate the fill similarity over the image
    '''
    bbsize = (
        (max(r1['max_x'], r2['max_x']) - min(r1['min_x'], r2['min_x']))
        * (max(r1['max_y'], r2['max_y']) - min(r1['min_y'], r2['min_y']))
    )
    return 1.0 - (bbsize - r1['size'] - r2['size']) / imsize


def _calc_sim(r1: dict, r2: dict, imsize: int) -> np.float64:
    return (_sim_color(r1, r2) + _sim_texture(r1, r2)
            + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize))


def _calc_color_hist(img: np.ndarray, color_bins: int) -> np.ndarray:
    '''
    Calculate color histogram for each region

    The size of output histogram will be BINS * COLOR_CHANNELS(3)

    Number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]

    Extract HSV
    '''

    hist = np.array([])

    for color_channel in (0, 1, 2):
        # extracting one color channel
        c = img[:, color_channel]

        # calculate histogram for each color and join to the result
        hist = np.concatenate(
            [hist] + [np.histogram(c, bins=color_bins, range=(0.0, 255.0))[0]])

    # L1 normalize
    hist = hist / len(img)

    return hist


def _calc_texture_gradient(img) -> np.ndarray:
    '''
    Calculate texture gradient for entire image

    The original SelectiveSearch algorithm proposed Gaussian derivative
    for 8 orientations, but we use LBP instead.

    Output will be [height(*)][width(*)]
    '''
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for color_channel in (0, 1, 2):
        ret[:, :, color_channel] = feature.local_binary_pattern(
            img[:, :, color_channel], 8, 1.0)

    return ret


def _calc_texture_hist(img: np.ndarray, texture_bins: int) -> np.ndarray:
    '''
    Calculate texture histogram for each region

    Calculate the histogram of gradient for each colors

    The size of output histogram will be
        BINS * ORIENTATIONS * COLOR_CHANNELS(3)
    '''

    hist = np.array([])

    for color_channel in (0, 1, 2):
        # mask by the color channel
        fd = img[:, color_channel]

        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = np.concatenate(
            [hist] + [np.histogram(fd, bins=texture_bins, range=(0.0, 255.0))[0]])

    # L1 Normalize
    hist = hist / len(img)

    return hist


def _extract_regions(img: np.ndarray, color_bins: int, texture_bins: int) -> dict:
    R: dict = {}

    # get hsv image
    hsv = color.rgb2hsv(img[:, :, :3])

    # pass 1: count pixel positions
    for y, i in enumerate(img):
        for x, (r, g, b, l) in enumerate(i):
            # initialize a new region
            if l not in R:
                R[l] = {
                    'min_x': 0xffff, 'min_y': 0xffff,
                    'max_x': 0, 'max_y': 0, 'labels': [l]}

            # bounding box
            if R[l]['min_x'] > x:
                R[l]['min_x'] = x
            if R[l]['min_y'] > y:
                R[l]['min_y'] = y
            if R[l]['max_x'] < x:
                R[l]['max_x'] = x
            if R[l]['max_y'] < y:
                R[l]['max_y'] = y

    # pass 2: calculate texture gradient
    tex_grad = _calc_texture_gradient(img)

    # pass 3: calculate color histogram of each region
    for k, v in R.items():
        # color histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]['size'] = len(masked_pixels / 4)
        R[k]['hist_c'] = _calc_color_hist(masked_pixels, color_bins)

        # texture histogram
        R[k]['hist_t'] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k], texture_bins)

    return R


def _extract_neighbors(regions: dict) -> list:

    def intersect(a: dict, b: dict) -> bool:
        if (a['min_x'] < b['min_x'] < a['max_x']
                and a['min_y'] < b['min_y'] < a['max_y']) or (
            a['min_x'] < b['max_x'] < a['max_x']
                and a['min_y'] < b['max_y'] < a['max_y']) or (
            a['min_x'] < b['min_x'] < a['max_x']
                and a['min_y'] < b['max_y'] < a['max_y']) or (
            a['min_x'] < b['max_x'] < a['max_x']
                and a['min_y'] < b['min_y'] < a['max_y']):
            return True
        return False

    R = list(regions.items())
    neighbors = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbors.append((a, b))

    return neighbors


def _merge_regions(r1: dict, r2: dict) -> dict:
    new_size = r1['size'] + r2['size']
    rt = {
        'min_x': min(r1['min_x'], r2['min_x']),
        'min_y': min(r1['min_y'], r2['min_y']),
        'max_x': max(r1['max_x'], r2['max_x']),
        'max_y': max(r1['max_y'], r2['max_y']),
        'size': new_size,
        'hist_c': (
            r1['hist_c'] * r1['size'] + r2['hist_c'] * r2['size']) / new_size,
        'hist_t': (
            r1['hist_t'] * r1['size'] + r2['hist_t'] * r2['size']) / new_size,
        'labels': r1['labels'] + r2['labels']
    }
    return rt


def selective_search(im_orig: np.ndarray, scale: float=1.0, sigma: float=0.8, min_size: int=50,
                     color_bins: int=25, texture_bins: int=10) -> tuple:
    '''Selective Search

    Parameters
    ----------
        im_orig : ndarray
            Input image
        scale : int
            Free parameter. Higher means larger clusters in felzenszwalb segmentation.
        sigma : float
            Width of Gaussian kernel for felzenszwalb segmentation.
        min_size : int
            Minimum component size for felzenszwalb segmentation.
        color_bins : int
            Number of bins to be extracted when calculating the color histogram per region.
        texture_bins : int
            Number of bins to be extracted when calculating the texture histogram per region.
    Returns
    -------
        img : ndarray
            image with region label
            region label is stored in the 4th value of each pixel [r,g,b,(region)]
        regions : array of dict
            [
                {
                    'rect': (min_x, min_y, max_x, max_y),
                    'labels': [...],
                    'size': component_size
                },
                ...
            ]
    '''
    if im_orig.shape[2] != 3:
        raise ValueError('Expected 3-channel image')

    # load image and get smallest regions
    # region label is stored in the 4th value of each pixel [r,g,b,(region)]
    img = _generate_segments(im_orig, scale, sigma, min_size)

    if img is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]
    R = _extract_regions(img, color_bins, texture_bins)

    # extract neighboring information
    neighbors = _extract_neighbors(R)

    # calculate initial similarities
    S = {}
    for (ai, ar), (bi, br) in neighbors:
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    # hierarchal search
    while S != {}:
        # get highest similarity
        i, j = sorted(S.items(), key=lambda i: i[1])[-1][0]

        # merge corresponding regions
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])

        # mark similarities for regions to be removed
        key_to_delete = []
        for k, v in S.items():
            if (i in k) or (j in k):
                key_to_delete.append(k)

        # remove old similarities of related regions
        for k in key_to_delete:
            del S[k]

        # calculate similarity set with the new region
        for k in [a for a in key_to_delete if a != (i, j)]:
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)

    regions = []
    for k, r in R.items():
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'], r['max_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    return img, regions
