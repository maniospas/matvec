#  python setup.py build
#  python setup.py bdist_wheel
#  twine upload dist/*

from setuptools import setup, Extension, find_packages
from distutils.command.build_ext import build_ext as build_ext_orig

with open("README.md", "r") as file:
    long_description = file.read()


class build_ext(build_ext_orig):
    def build_extension(self, ext):
        self._ctypes = isinstance(ext, Extension)
        return super().build_extension(ext)

    def get_export_symbols(self, ext):
        if self._ctypes:
            return ext.export_symbols
        return super().get_export_symbols(ext)

    def get_ext_filename(self, ext_name):
        if self._ctypes:
            return ext_name + '.pyd'
        return super().get_ext_filename(ext_name)


setup(
    name='matvec',
    version='0.0.11',
    author="Emmanouil (Manios) Krasanakis",
    author_email="maniospas@hotmail.com",
    description="Fast matrix transforms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maniospas/matvec",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy"
    ],
    py_modules=["matvec"],
    ext_modules=[
        Extension(
            "matvec/matvec.py",
            ["matvec.cpp"],
              extra_compile_args=['-openmp'],
              extra_link_args=[],
        ),
    ],
    cmdclass={'build_ext': build_ext},
)