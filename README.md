# Selective Search Implementation for Python

This is an implementation of Selective Search for Python 3 as it can be found in http://cs.brown.edu/~pff/papers/seg-ijcv.pdf. The code of this repository is based on https://github.com/AlpacaDB/selectivesearch.

## Download

There is no package available to download, at the moment. You can however, download the implementation of Selective Search with the following command:
```
git clone https://github.com/vsakkas/selective-search.git
```

## Usage

The following example shows how you can use Selective Search, after you have downloaded and imported it in your project:

```python
import skimage.data
import selectivesearch

img = skimage.data.astronaut()
img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
regions[:10]
=>
[{'labels': [0.0], 'rect': (0, 0, 15, 24), 'size': 260},
 {'labels': [1.0], 'rect': (13, 0, 1, 12), 'size': 23},
 {'labels': [2.0], 'rect': (0, 15, 15, 11), 'size': 30},
 {'labels': [3.0], 'rect': (15, 14, 0, 0), 'size': 1},
 {'labels': [4.0], 'rect': (0, 0, 61, 153), 'size': 4927},
 {'labels': [5.0], 'rect': (0, 12, 61, 142), 'size': 177},
 {'labels': [6.0], 'rect': (7, 54, 6, 17), 'size': 8},
 {'labels': [7.0], 'rect': (28, 50, 18, 32), 'size': 22},
 {'labels': [8.0], 'rect': (2, 99, 7, 24), 'size': 24},
 {'labels': [9.0], 'rect': (14, 118, 79, 117), 'size': 4008}]
```

See also an example/example.py which generates :
![alt tag](https://github.com/AlpacaDB/selectivesearch/raw/develop/example/result.png)

## Parameters of Selective Search


#### im_orig : ndarray
 Input image
 
#### scale : int
 Free parameter. Higher means larger clusters in felzenszwalb segmentation.
 
#### sigma : float
 Width of Gaussian kernel for felzenszwalb segmentation.
 
#### min_size : int
 Minimum component size for felzenszwalb segmentation.
 
#### color_bins : int
 Number of bins to be extracted when calculating the color histogram per region.
 
#### texture_bins : int
 Number of bins to be extracted when calculating the texture histogram per region.
 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
