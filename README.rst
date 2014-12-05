===============================================================================
Besra - image classification for protein crystallization experiments
===============================================================================

-------------------------------------------------------------------------------
About
-------------------------------------------------------------------------------

Besra is a tool for auto-classifying protein crystallization experiments. Source
and binary releases are available on `GitHub <https://github.com/ubccr/besra/releases>`_.

The goal is to implement a fast and accurate binary classifier for determining
crystal-positive vs crystal-negative images in high-throughput protein
crystallization experiments. Accuracy equal or better to that of a human would
be considered a success. Current methods take upwards of ~10 hours to classify
1536 images. Speeding up the classification will allow better integration into
existing expert knowledge systems/pipelines and enable robust evaluation and
tuning of classification algorithms across millions of images. Extending from
binary classification to n-way classification (clear, precipitate, skin, phase
separation) is supported however performance has not been extensivley tested.

-------------------------------------------------------------------------------
Quickstart
-------------------------------------------------------------------------------

First need to train on a set of images. besra-trainer requires a <TAB> separated
input file of image paths and class labels. For example, an input file with
crystal-positive = 1 and crystal-negative = 0 looks like this::

  /images/png/X0000051270868200506241635.png    1
  /images/png/X0000049750501200505270943.png    1
  /images/png/X0000049151305200506061345.png    1
  /images/png/X0000050511419200507041505.png    0
  /images/png/X0000051830553200507012227.png    0
  /images/png/X0000050611108200507051159.png    0
  ...

The path should be the full path to the image on the filesystem and the class
label should be a float. The input file requires at least 2 distinct class
labels. To train a set of images run::

  $ besra-trainer -i input.tsv -v

To speed up processing set --threads option equal to the number of cores
available on your machine. For the full set of options see::

  $ besra-trainer --help

This command will output 2 files: stats-model.xml and bow-vocab.yml which can
later be used to classify images (without having to re-train each time).

To classify a directory of images::

  $ besra-classify -i /path/to/images -m stats-model.xml -b bow-vocab.yml -v

To classify images using an input file (must be one image path per line,
similar to the input for besra-trainer)::

  $ besra-classify -i input.tsv -m stats-model.xml -b bow-vocab.yml -v

Results are written to a file named: besra-results.tsv

-------------------------------------------------------------------------------
Implementation
-------------------------------------------------------------------------------

Besra currently uses the bag-of-visual-words method [1] and a support vector
machine (SVM) classifier. Keypoints/local features are computed from the
training set using SURF [2] descriptors and clustered using k-means into a
visual vocabulary. An SVM with a linear kernel is used for image
classification.

The assumption is that the clustered features computed from crystal-positive
images will be distinct enough from crystal-negative images to produce an
accurate classifier. 

TODO:

- Optimize the parameters of SURF (hessian threshold, gaussian pyramid
  octaves, etc.). What are the appropriate settings for our data?

- Optimize the number of k-means clusters when computing the BOW
  vocabulary. Is there an optimal number of clusters?

- Experiment with other descriptor/keypoint extractors/detectors available in
  OpenCV (FAST, MSER, ORB, BRISK, etc.). See `features2d <http://docs.opencv.org/modules/features2d/doc/features2d.html>`_ 
  for the complete list.

- Experiment with different SVM types and kernels. See `svm <http://docs.opencv.org/modules/ml/doc/support_vector_machines.html>`_

- Test performance on other classes of images (clear, precipitate, phase separation). 

- Test OpenMP threads vs SURF_GPU

-------------------------------------------------------------------------------
Requirements
-------------------------------------------------------------------------------

- `boost <http://www.boost.org/>`_ >= 1.57
- `OpenCV <http://opencv.org/>`_ >= 2.4.9
- `CMake <http://www.cmake.org/>`_ >= 1.8

-------------------------------------------------------------------------------
Installation
-------------------------------------------------------------------------------

Besra uses cmake. To compile run::

  $ git clone https://github.com/ubccr/besra.git besra
  $ cd besra
  $ mkdir build
  $ cd build
  $ cmake ..
  $ make

If boost is compiled in a non-standard location run::

  $ BOOST_ROOT=/path/to/boost cmake -DBoost_NO_SYSTEM_PATHS=TRUE ..

If OpenCV is compiled in a non-standard location run::

  $ OpenCV_DIR=/path/to/opencv cmake ..

To compile besra with GPU support (requires GPU/CUDA support to be compiled in
OpenCV)::

  $ cmake -DUSE_GPU=on ..

-------------------------------------------------------------------------------
Compiling OpenCV
-------------------------------------------------------------------------------

To enable multi-threaded clustering, compile OpenCV with OpenMP support. For
example::

  $ unzip opencv-2.4.x.zip
  $ cd opencv-2.4.x/
  $ mkdir build
  $ cd build
  $ cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/path/to/localdir -DWITH_OPENMP=Yes ..

For more info see `OpenCV docs <http://docs.opencv.org/trunk/doc/tutorials/introduction/linux_install/linux_install.html>`_.

-------------------------------------------------------------------------------
Compiling Boost
-------------------------------------------------------------------------------

To compile boost::

  $ tar xvf boost-1.xx.x.tar.gz
  $ cd boost_1_xx_x
  $ ./bootstrap.sh --prefix=/path/to/localdir \
       --with-libraries=log,thread,date_time,filesystem,system,program_options
  $ ./b2 install

For more info see `boost docs <http://www.boost.org/doc/libs/1_57_0/more/getting_started/unix-variants.html>`_.

-------------------------------------------------------------------------------
License
-------------------------------------------------------------------------------

Copyright (C) 2014 Andrew E. Bruno

Besra is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <http://www.gnu.org/licenses/>.

-------------------------------------------------------------------------------
References
-------------------------------------------------------------------------------

[1] Csurka, Gabriella, et al. "Visual categorization with bags of keypoints."
    Workshop on statistical learning in computer vision, ECCV. Vol. 1. No. 1-22.
    2004.

[2] Bay, H. and Tuytelaars, T. and Van Gool, L. "SURF: Speeded Up Robust
    Features", 9th European Conference on Computer Vision, 2006
