# Selective Search Implementation for Python 3

This is an implementation of Selective Search for Python 3 as it can be found in http://cs.brown.edu/~pff/papers/seg-ijcv.pdf. The code of this repository is based on https://github.com/AlpacaDB/selectivesearch.

## Requirements

Running this program on your machine requires:
* `Python 3.6 (or newer)`
* `numpy`
* `skimage`

## Download

There is no package available to download at the moment. You can however, download the implementation of Selective Search with the following command:
```
git clone https://github.com/vsakkas/selective-search.git
```

## Usage

The following example shows how you can use Selective Search, after you have downloaded this implementation and imported it in your own project:

```python
img = skimage.data.astronaut()
img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
regions[:5]
=>
[{'rect': (0, 0, 15, 26),    'size': 313,   'labels': [0.0]},
 {'rect': (0, 0, 61, 144),   'size': 4794,  'labels': [1.0]},
 {'rect': (10, 0, 113, 227), 'size': 2071,  'labels': [2.0]},
 {'rect': (46, 0, 399, 326), 'size': 43655, 'labels': [3.0]}
 {'rect': (396, 0, 408, 74), 'size': 193,   'labels': [4.0]}]
```

You can also check [example.py](https://github.com/vsakkas/selective-search/blob/master/example/example.py) which generates the following image:
![alt tag](https://github.com/AlpacaDB/selectivesearch/raw/develop/example/result.png)

## Parameters


#### im_orig : ndarray
 ```
 Input image
 ```
 
#### scale : int
 ```
 Free parameter. Higher means larger clusters in felzenszwalb segmentation.
 ```
 
#### sigma : float
 ```
 Width of Gaussian kernel for felzenszwalb segmentation.
 ```
 
#### min_size : int
 ```
 Minimum component size for felzenszwalb segmentation.
 ```
 
#### color_bins : int
 ```
 Number of bins to be extracted when calculating the color histogram per region.
 ```
 
#### texture_bins : int
 ```
 Number of bins to be extracted when calculating the texture histogram per region.
 ```
 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
