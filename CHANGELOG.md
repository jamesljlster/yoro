# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

## \[Unreleased\]

## \[0.2.1\] - 2021-04-19

The release is for PyTorch 1.8 compatibility and small improvement on training utility.

-   Training units:

    Training units reduces iterating cost by concatenating several epochs into one iteration.  
    The parameter can be set in `train_param / train_units`.

    ``` yaml
    train_param:
      train_units: 0
    ```

    The setting values and corresponding behaviors for train\_units is shown below:

    -   0: Auto selecting by `min(esti_epoch, bak_epoch)`.
    -   &gt; 0: Using this value.

    By default, train\_units will be zero if it is not presented in configuration file.

-   Moving average for training informations:

    Moving factor can be controlled by `train_param / moving_factor`.

    ``` yaml
    train_param:
      moving_factor: 0.01
    ```

    By default, moving\_factor will be 0.01 if it is not presented in configuration file.

### Added

-   Iterating cost can be reduced by training units.

### Changed

-   Training information is now presented by reduced by moving average.
-   mAP evaluation will now punish duplicated detection.
-   Transform functions is refactored with TorchVision 0.9.

### Fixed

-   C++ API compatibility with PyTorch 1.8.

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
