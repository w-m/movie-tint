movie-tint
==========

Render the primary colors of movies over time into beautiful images.

The results are over at http://movietint.tumblr.com


Requirements
============

* Python 2.7
* OpenCV+ffmpeg with Python bindings
* numpy
* py-progressbar

Usage
=====

Call with an input movie file and output file name, e.g. `python movie-tint.py my-movie.avi my-tint.png`

Parameters for the color extraction can be set in the header of the script file.


Algorithm
=========

The frames of the movie are scaled down and split into chunks. Each chunk will account for one pixel horizontically in the resulting image, the chunk size can be specified with FRAMES_PER_PIXEL.

The pixels are converted to the HSV color space. The hue of all pixels is shifted by 60 degrees to bring the different shades of red together.

The primary colors of one chunk are found with a k-means-clustering with a k of 3.

The cluster centers are drawn into the resulting image; the more pixels belong to a cluster, the bigger the circle.

