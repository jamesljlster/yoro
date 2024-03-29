from os.path import join, dirname


def get_config(config_file):
    return join(dirname(__file__), config_file)


def yolov3():
    return get_config('yolov3.cfg')


def yolov3_tiny():
    return get_config('yolov3-tiny.cfg')


def yolov4():
    return get_config('yolov4.cfg')


def yolov4_csp():
    return get_config('yolov4-csp.cfg')


def yolov4_tiny():
    return get_config('yolov4-tiny.cfg')
