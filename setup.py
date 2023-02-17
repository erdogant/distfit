from packaging import version
import setuptools
import re

# versioning ------------
VERSIONFILE="distfit/__init__.py"
getversion = re.search( r"^__version__ = ['\"]([^'\"]*)['\"]", open(VERSIONFILE, "rt").read(), re.M)
if getversion:
    new_version = getversion.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

import matplotlib
if version.parse(matplotlib.__version__) < version.parse('3.5.2'):
    raise ImportError(
        'This release requires matplotlib version >= 3.5.2. Try: pip install -U matplotlib')

# Setup ------------
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     install_requires=['packaging', 'matplotlib>=3.5.2','numpy','pandas','tqdm','statsmodels','scipy','pypickle', 'colourmap>=1.1.10'],
     python_requires='>=3',
     name='distfit',
     version=new_version,
     author="Erdogan Taskesen",
     author_email="erdogant@gmail.com",
     description="distfit is a python library for probability density fitting.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://erdogant.github.io/distfit",
	 download_url = 'https://github.com/erdogant/distfit/archive/'+new_version+'.tar.gz',
     packages=setuptools.find_packages(), # Searches throughout all dirs for files to include
     include_package_data=True, # Must be true to include files depicted in MANIFEST.in
     license_files=["LICENSE"],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
