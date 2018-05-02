import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    import sys
    sys.path.insert(0, '../selectivesearch/')
    import selectivesearch

    # loading astronaut image
    img = skimage.data.astronaut()

    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    print(regions[:5])

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        min_x, min_y, max_x, max_y = r['rect']
        width, height = max_x - min_x, max_y - min_y
        if width / height > 1.3 or height / width > 1.3:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for min_x, min_y, max_x, max_y in candidates:
        print(min_x, min_y, max_x, max_y)
        width, height = max_x - min_x, max_y - min_y
        rect = mpatches.Rectangle(
            (min_x, min_y), width, height, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    main()
