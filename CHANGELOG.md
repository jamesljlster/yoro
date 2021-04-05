# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

## \[Unreleased\]

### Changed

-   mAP evaluation will now punish duplicated detection.

### Fixed

-   C++ API compatibility with PyTorch 1.8.0.

## \[0.2.0\] - 2021-03-14

### Added

-   Add mAP estimation in validation progress for YORO training,
    and map\_evaluator utility for model performance evaluation.

### Changed

-   NMS is now filtering prediction with (objectness \* class confidence)
    instead of only objectness.
    This change breaks backward compatibility of torchscript model.
-   In order to support model with different value for width and height,
    anchor\_cluster is now using padding to aspect ratio instead of padding
    to square.

### Fixed

-   Wrong description of ‘height’ command line parameter for anchor\_cluster.
-   Add implicit build dependency ‘cudnn’ in readme.
-   Incorrect rotated bounding box similarity calculation.

## \[0.1.0\] - 2021-01-10

Initial release.

  [Keep a Changelog]: https://keepachangelog.com/en/1.0.0/
  [Semantic Versioning]: https://semver.org/spec/v2.0.0.html
