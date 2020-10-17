import sys
import distutils.sysconfig

import os
from os import makedirs, environ
from os.path import abspath, exists, join

from subprocess import check_call
from setuptools import setup, find_packages

import torch


class CMakeBuild(object):

    def __init__(self, source, cmake_args={},
                 extra_incpath=[], extra_ldpath=[]):

        # Find source and build path
        srcDir = abspath(source)
        buildDir = join(srcDir, 'build')

        # Convert cmake args
        args = [
            '-D%s=%s' % (arg, cmake_args[arg]) for arg in cmake_args.keys()
        ]

        # Prefix paths
        args += [
            '-DCMAKE_PREFIX_PATH=%s' % distutils.sysconfig.PREFIX
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


if __name__ == '__main__':

    # Build yoro_api
    torchPath = torch.__path__[0]
    CMakeBuild(
        source='.',
        cmake_args={
            'Python_ROOT_DIR': distutils.sysconfig.PREFIX,
            'Python_FIND_STRATEGY': 'LOCATION',
            'Python_FIND_REGISTRY': 'FIRST',
            'Torch_DIR': join(torchPath, 'share/cmake/Torch'),
            'Caffe2_DIR': join(torchPath, 'share/cmake/Caffe2')
        },
        extra_ldpath=[join(torchPath, 'lib')]
    )

    # Setup
    setup(

        # Basic informations
        name='yoro',
        version='0.1.0',
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
                'bin/*',
                'include/yoro_api/*',
                'lib/cmake/yoro_api/*',
                'lib/yoro_api/*'
            ]
        }
    )
