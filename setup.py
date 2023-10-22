from os import path, environ
from distutils.core import Extension, setup
from Cython.Build import cythonize

VERSION = "0.3.3"
LICENSE = "MIT"
DESCRIPTION = "Python wrapper for quda written in Cython."

ld_library_path = [path.abspath(_path) for _path in environ["LD_LIBRARY_PATH"].strip().split(":")]

for libquda_path in ld_library_path:
    if path.exists(path.join(libquda_path, "libquda.so")):
        break
else:
    raise RuntimeError("Cannot find libquda.so in LD_LIBRARY_PATH environment")

BUILD_QCU = False
for libqcu_path in ld_library_path:
    if path.exists(path.join(libqcu_path, "libqcu.so")):
        BUILD_QCU = True
        break
else:
    import warnings

    warnings.warn("Cannot find libqcu.so in LD_LIBRARY_PATH environment", RuntimeWarning)

extensions = [
    Extension(
        name="pyquda.pointer",
        sources=["pyquda/src/pointer.pyx"],
        language="c",
    ),
    Extension(
        name="pyquda.pyquda",
        sources=["pyquda/src/pyquda.pyx"],
        include_dirs=["pyquda/include/quda"],
        library_dirs=[libquda_path],
        libraries=["quda"],
        language="c",
    ),
    Extension(
        name="pyquda.malloc_pyquda",
        sources=["pyquda/src/malloc_pyquda.pyx"],
        include_dirs=["pyquda/include/quda"],
        library_dirs=[libquda_path],
        libraries=["quda"],
        language="c++",
    ),
]

if BUILD_QCU:
    extensions.append(
        Extension(
            name="pyquda.pyqcu",
            sources=["pyquda/src/pyqcu.pyx"],
            include_dirs=["pyquda/include/qcu"],
            library_dirs=[libqcu_path],
            libraries=["qcu"],
            language="c",
        )
    )

packages = [
    "pyquda",
    "pyquda.dslash",
    "pyquda.utils",
]
package_dir = {
    "pyquda": "pyquda",
}
package_data = {
    "pyquda": ["*.pyi", "*.pxd", "src.pxd", "include/**"],
}

setup(
    name="PyQuda",
    version=VERSION,
    description=DESCRIPTION,
    author="SaltyChiang",
    author_email="SaltyChiang@users.noreply.github.com",
    packages=packages,
    ext_modules=cythonize(extensions, language_level="3"),
    license=LICENSE,
    package_dir=package_dir,
    package_data=package_data,
)
