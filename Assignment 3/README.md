# visionProcessor-CMPE537

Computer Vision

This repository holds an enhanced version of the original, focusing on Local Descriptor based Image Classification.

Please see the CMPE_537_Assignment_3_Report.pdf in this folder for original detailed report.

## Installation

    $ pip install -r requirements.txt

You can [Download the Caltech 101 dataset](https://drive.google.com/drive/folders/1hwfdetoGWqt0jdePmQvLska1ccEeBHOe?usp=sharing) and add this to `Dataset/Caltech20`.

Results are saved to the `Data/` folder.

## Image Classification Pipeline

- Compute local descriptors

  - SIFT
  - HOG (own implementation)
  - Local Binary Patterns (own implementation)
  - SURF
  - ORB
  - CenSure
  - FAST
  - BR