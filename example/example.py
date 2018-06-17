import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main():
    # Workaround to import selectivesearch from its directory
    import sys
    sys.path.insert(0, '../selectivesearch/')
    import selectivesearch

    # Load astronaut image
    img = skimage.data.astronaut()

    # Perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)

    print(regions[:5])

    candidates = set()
    for r in regions:
        # Exclude same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # Exclude small regions
        if r['size'] < 1000:
            continue
        # Exclude distorted rects
        min_x, min_y, max_x, max_y = r['rect']
        width, height = max_x - min_x + 1, max_y - min_y + 1
        if width / height > 1.5 or height / width > 1.5:
            continue
        candidates.add(r['rect'])

    # Draw rectangles on the original image and display it
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for min_x, min_y, max_x, max_y in candidates:
        print(min_x, min_y, max_x, max_y)
        width, height = max_x - min_x + 1, max_y - min_y + 1
        rect = mpatches.Rectangle(
            (min_x, min_y), width, height, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()


if __name__ == '__main__':
    main()
