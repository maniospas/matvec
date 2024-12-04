from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


ext_modules = [
    Extension(
        'matvec',
        ['matvec.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-std=c++17', '-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]

setup(
    name='matvec',
    version='0.2.0',
    author="Emmanouil (Manios) Krasanakis",
    author_email="maniospas@hotmail.com",
    description="Fast matrix transforms",
    url="https://github.com/maniospas/matvec",
    ext_modules=ext_modules,
    install_requires=['numpy', 'pybind11'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
