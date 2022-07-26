from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='sinkhorn',
      version='0.1',
      ext_modules=[cpp_extension.CppExtension('sinkhorn', ['sinkhorn.cpp'])],
      author='HENRI DE PLAEN',
      author_email='henri.deplaen@esat.kuleuven.be',
      cmdclass={'build_ext': cpp_extension.BuildExtension})