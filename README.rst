===============================================================================
Besra - image classification for protein crystallization experiments
===============================================================================

-------------------------------------------------------------------------------
About
-------------------------------------------------------------------------------

Besra is a tool for auto-classifying protein crystallization experiments. Source
and binary releases are available on `GitHub <https://github.com/ubccr/besra/releases>`_.

-------------------------------------------------------------------------------
Quickstart
-------------------------------------------------------------------------------

First need to train on a set of images. besra-trainer requires a <TAB> separated
input file of image paths and class labels. For example, an input file with
crystal = 1 and no-crystal = 0 looks like this::

  /images/png/X0000051270868200506241635.png    1
  /images/png/X0000049750501200505270943.png    1
  /images/png/X0000049151305200506061345.png    1
  /images/png/X0000050511419200507041505.png    0
  /images/png/X0000051830553200507012227.png    0
  /images/png/X0000050611108200507051159.png    0
  ...

The path should be the full path to the image on the filesystem and the class
label needs to be a float. The input file needs to have at last 2 distinct class
labels. To train a set of images run::

  $ besra-trainer -i input.tsv -v

For the full set of options see::

  $ besra-trainer --help

This command will output 2 files: stats-model.xml and bow-vocab.yml which can
later be used to classify images (without having to re-train each time).

To classify a directory of images::

  $ besra-classify -i /path/to/images -m stats-model.xml -b bow-vocab.yml -v

To classify images using an input file (must be one image path per line,
similiar to the input for besra-trainer)::

  $ besra-classify -i input.tsv -m stats-model.xml -b bow-vocab.yml -v

Results are written to a file named: besra-results.tsv

-------------------------------------------------------------------------------
Requirements
-------------------------------------------------------------------------------

- `boost <http://www.boost.org/>`_ >= 1.57
- `OpenCV <http://opencv.org/>`_ >= 2.4.9

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
