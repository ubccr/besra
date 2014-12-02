===============================================================================
Besra - image classification for protein crystallization experiments
===============================================================================

------------------------------------------------------------------------
About
------------------------------------------------------------------------

Besra is a tool for auto-classifying protein crystallization experiments. The
classifier is currently binary (crystal/no-crystal).

------------------------------------------------------------------------
Requirments
------------------------------------------------------------------------

- boost >= 1.55
- OpenCV 2.4.x

------------------------------------------------------------------------
Installation
------------------------------------------------------------------------

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

To enable multi-threaded clustering, compile OpenCV with OpenMP support. For
example::

  $ unzip opencv-2.4.x.zip
  $ cd opencv-2.4.x/
  $ mkdir build
  $ cd build
  $ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/path/to/localdir -DWITH_OPENMP=Yes ..

------------------------------------------------------------------------
Usage
------------------------------------------------------------------------

First need to train on a set of positive/negative images::

  $ besra-trainer -p /path/to/crystal -n /path/to/nocrystal

This creates 2 files: stats-model.xml and bow-vocab.yml

To classify a directory of images::

  $ besra-classify -i /path/to/images -m stats-model.xml -v bow-vocab.yml

Results are written to a file named: besra-results.tsv

------------------------------------------------------------------------
License
------------------------------------------------------------------------

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
