# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

## \[Unreleased\]

### Fixed

-   Fix training mode error handling for utilities.

## \[0.4.0\] - 2021-08-09

The release is for degree encoding method update, with some new features introduced.

-   Learning scheduler:

    Add `steps` learning rate scheduler which is documented in:  
    <https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-%5Bnet%5D-section>

    If `lr_scheduler` is not specified, it is disabled by default.

    ``` yaml
    train_param:
      lr_scheduler:
        name: 'yoro.optim.lr_scheduler.steps'
        args:
          steps: [2500, 10000]
          scales: [0.1, 0.1]
          burnin_iters: 1000
    ```

-   Norm gradient clipper:

    Add gradient clipper mechanism to prevent gradient explosion:  
    <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>

    The `grad_clip_norm` setting in configuration file is mapping to the `max_norm`
    parameter of `torch.nn.utils.clip_grad_norm_`.

    ``` yaml
    train_param:
      grad_clip_norm: 100
    ```

### Added

-   Add out of range warning for degress groundtruth.

### Changed

-   Degree anchor is replaced by spaced constant numbers.  
    This change breaks backward compatibility of TorchScript model.
-   The “cls” training infomation of YORO is changed to confidence instead of accuracy.
-   Sample size of a training unit is now aligned by batch size for YORO model training.

### Fixed

-   Fix TorchScript annotation for YOROLayer.

## \[0.3.0\] - 2021-07-26

The release is mainly for multi-head support, some backbones from Darknet,
and model performance improvements.  
Backward compatibility is broken, retrain model with/without
pretrained weights is required.

-   Multi-head support:

    Anchor setting in configuration file is now changed to 3-dim float array,
    and the definition of anchor shape is (num_heads, num_anchors, 2).  
    Backbone output type for YORO mode is now changed to list of tensor.  
    Design anchors corresponding to backbone outputs is advised.

    In training progress,
    a ground truth bounding box can be matched by several anchors by exclusive matching.  
    Use `anchor_thresh` and `anchor_max_count` to control IoU matching threshold,
    and the maximum amount of anchors can be matched by a ground truth.

    ``` yaml
    train_param:
      anchor_thresh: 0.5
      anchor_max_count: 3
    ```

-   Backbones from Darknet:

    `yoro.backbone` module is renamed to `yoro.backbones`.  
    And the following backbones are implemented:

    -   YOLOv3: `yoro.backbones.darknet.YOLOv3`
    -   YOLOv3-Tiny: `yoro.backbones.darknet.YOLOv3_Tiny`
    -   YOLOv4: `yoro.backbones.darknet.YOLOv4`
    -   YOLOv4-CSP: `yoro.backbones.darknet.YOLOv4_CSP`
    -   YOLOv4-Tiny: `yoro.backbones.darknet.YOLOv4_Tiny`

-   Performance improvements:

    Bounding box regression is now using CIoU loss instead of MSE loss.  
    And objectness loss reduction is changed from mean to sum,
    this change reduces false positive detections.

    In order to balance between loss components, normalizers is adapted in this update.  
    If any loss normalizer setting is not given, default value 1.0 is used.

    ``` yaml
    train_param:
      loss_normalizer:
        obj: 1.5
        no_obj: 0.3
        class: 1.0
        box: 1.3
        degree_part: 1.0
        degree_shift: 1.0
    ```

-   Evaluation on multiple similarity threshold:

    `sim_threshold` now can be set with \[start, end, step\].  
    If scalar value is given, single threshold evaluation is performed.

    ``` yaml
    train_param:
      evaluator:
        conf_threshold: 0.5
        nms_threshold: 0.6
        sim_threshold: [0.5, 0.95, 0.05]
    ```

### Added

-   Add subdivision training support.

    ``` yaml
    train_param:
      subdivision: 2
    ```

### Changed

-   Desire network width and height are now required parameters for anchor_cluster.
-   map_evaluator is now use \[start, end, step\] as similarity threshold setting.
-   Bounding boxes encoding is changed to the same as YOLOv4:  
    <https://github.com/WongKinYiu/ScaledYOLOv4/issues/90>

### Fixed

-   Fix non-contiguous input data is passed in map_evaluator.

## \[0.2.2\] - 2021-04-26

Small fix for package building.  
CMake will now auto searching for required torch libraries.

### Fixed

-   Fix torch library searching under CMake build system.

## \[0.2.1\] - 2021-04-19

The release is for PyTorch 1.8 compatibility and small improvement on training utility.

-   Training units:

    Training units reduces iterating cost by concatenating several epochs into one iteration.  
    The parameter can be set in `train_param / train_units`.

    ``` yaml
    train_param:
      train_units: 0
    ```

    The setting values and corresponding behaviors for train_units is shown below:

    -   0: Auto selecting by `min(esti_epoch, bak_epoch)`.
    -   \> 0: Using this value.

    By default, train_units will be zero if it is not presented in configuration file.

-   Moving average for training information:

    Moving factor can be controlled by `train_param / moving_factor`.

    ``` yaml
    train_param:
      moving_factor: 0.01
    ```

    By default, moving_factor will be 0.01 if it is not presented in configuration file.

### Added

-   Iterating cost can be reduced by training units.

### Changed

-   Training information is now presented by reduced by moving average.
-   mAP evaluation will now punish duplicated detection.
-   Transform functions are refactored with TorchVision 0.9.

### Fixed

-   C++ API compatibility with PyTorch 1.8.

## \[0.2.0\] - 2021-03-14

### Added

-   Add mAP estimation in validation progress for YORO training,
    and map_evaluator utility for model performance evaluation.

### Changed

-   NMS is now filtering prediction with (objectness \* class confidence)
    instead of only objectness.
    This change breaks backward compatibility of torchscript model.
-   In order to support model with different value for width and height,
    anchor_cluster is now using padding to aspect ratio instead of padding
    to square.

### Fixed

-   Wrong description of ‘height’ command line parameter for anchor_cluster.
-   Add implicit build dependency ‘cudnn’ in readme.
-   Incorrect rotated bounding box similarity calculation.

## \[0.1.0\] - 2021-01-10

Initial release.

  [Keep a Changelog]: https://keepachangelog.com/en/1.0.0/
  [Semantic Versioning]: https://semver.org/spec/v2.0.0.html
