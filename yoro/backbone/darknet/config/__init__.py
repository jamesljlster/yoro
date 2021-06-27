from os.path import join, dirname


def get_config(config_file):
    return join(dirname(__file__), config_file)


def yolov3():
    return get_config('yolov3.cfg')
