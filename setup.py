import sys
from importlib import import_module

import os
from os import makedirs, environ
from os.path import abspath, exists, join, normpath

import subprocess
from subprocess import check_call
from setuptools import setup, find_packages, dist


def env_prefix():
    return normpath(sys.prefix)


class CMakeBuild(object):

    def __init__(self, source, cmake_args={},
                 extra_incpath=[], extra_ldpath=[]):

        # Find source and build path
        srcDir = abspath(source)
        buildDir = join(srcDir, 'py_build')

        # Convert cmake args
        args = [
            '-D%s=%s' % (arg, cmake_args[arg]) for arg in cmake_args.keys()
        ]

        # Prefix paths
        args += [
            '-DCMAKE_PREFIX_PATH=%s' % env_prefix()
        ]

        # Build type
        if '--debug' in sys.argv:
            args.append('-DCMAKE_BUILD_TYPE=Debug')

        # Set environment variables
        incflags = ' '.join(['-I%s' % inc for inc in extra_incpath])
        ldflags = ' '.join(['-L%s' % ld for ld in extra_ldpath])

        env = environ.copy()
        env['CXXFLAGS'] = ' '.join([env.get('CXXFLAGS', ''), incflags])
        env['LDFLAGS'] = ' '.join([env.get('LDFLAGS', ''), ldflags])

        # Build project
        if not exists(buildDir):
            makedirs(buildDir)

        check_call(['cmake', srcDir] + args, cwd=buildDir, env=env)
        check_call([
            'cmake',
            '--build', '.',
            '--target', 'install',
            '--parallel', str(os.cpu_count())
        ], cwd=buildDir)


def build_yoro_api():

    # Get torch package
    try:
        torch = import_module('torch')
    except:
        raise RuntimeError('PyTorch package is not found')

    # Build yoro_api
    torchPath = torch.__path__[0]
    CMakeBuild(
        source='.',
        cmake_args={
            'Python_ROOT_DIR': env_prefix(),
            'Python_FIND_STRATEGY': 'LOCATION',
            'Python_FIND_REGISTRY': 'FIRST',
            'Torch_DIR': join(torchPath, 'share/cmake/Torch'),
            'Caffe2_DIR': join(torchPath, 'share/cmake/Caffe2')
        },
        extra_ldpath=[join(torchPath, 'lib')]
    )


if __name__ == '__main__':

    # Fetch build dependencies
    dist.Distribution().fetch_build_eggs([
        'torch'
    ])

    # Build YORO API
    build_yoro_api()

    # Resolve dependencies
    install_requires = [
        'torch',
        'torchvision',
        'numpy',
        'pyyaml',
        'tqdm'
    ]

    try:
        import_module('cv2')
    except:
        install_requires += ['opencv-python']

    # Setup
    setup(

        # Basic informations
        name='yoro',
        version='0.2.0',
        description='YORO: A YOLO Variant for Rotated Object Detection',
        url='https://gitlab.ical.tw/jamesljlster/yoro',
        author='Cheng-Ling Lai',
        author_email='jamesljlster@gmail.com',
        license='GPLv3',

        # Packaging
        packages=find_packages(exclude=('test', 'utils')),
        package_data={
            'yoro': [
                'api/*.so',
                'include/yoro_api/*',
                'lib/cmake/yoro_api/*',
                'lib/yoro_api/*'
            ]
        },
        scripts=[
            'yoro/bin/anchor_cluster',
            'yoro/bin/backup_exporter',
            'yoro/bin/recaller',
            'yoro/bin/trainer',
            'yoro/bin/pretrain_exporter',
            'yoro/bin/map_evaluator'
        ],

        # Dependencies
        install_requires=install_requires
    )
