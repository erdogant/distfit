
Coding quality
'''''''''''''''''''''

Higher quality software has fewer defects, better security, and better performance, which leads to happier users who can work more effectively.
Code reviews are an effective method for improving software quality. McConnell (2004) suggests that unit testing finds approximately 25% of defects, function testing 35%, integration testing 45%, and code review 55-60%. 
While this means that none of these methods is good enough on their own and that they should be combined, clearly code review is an essential tool here.

This library is therefore developed with several techniques, such as code styling, low complexity, docstrings, reviews, and unit tests.
Such conventions are helpfull to improve the quality, make the code cleaner and more understandable but alos to trace future bugs, and spot syntax errors.


library
-------

The file structure of the generated package looks like:


.. code-block:: bash

    path/to/distfit/
    ├── .gitignore
    ├── docs
    │   ├── conf.py
    │   ├── index.rst
    │   └── ...
    ├── LICENSE
    ├── MANIFEST.in
    ├── distfit
    │   ├── __init__.py
    │   ├── __version__.py
    │   └── distfit.py
    ├── README.md
    ├── requirements.txt
    ├── setup.py
    └── tests
        ├── __init__.py
        └── test_distfit.py


Style
-----

This library is compliant with the PEP-8 standards.
PEP stands for Python Enhancement Proposal and sets a baseline for the readability of Python code.
Each public function contains a docstring that is based on numpy standards.
    

Complexity
----------

This library has been developed by using measures that help decreasing technical debt.
Version 1.1.0 of the ``distfit`` library scored, according the code analyzer: **2.25**, for which values > 0 are good and 10 is a maximum score.
Developing software with low(er) technical depth (a higher code analyser score) may take extra development time, but has many advantages:

* Higher quality code
* easier maintanable
* Less prone to bugs and errors
* Higher security


Unit tests
----------

The use of unit tests is essential to garantee a consistent output of developed functions.
The following tests are secured using :func:`tests.test_distfit`:

* The input are checked.
* The output values are checked and whether they are encoded properly.
* The check of whether parameters are handled correctly.


.. code-block:: bash

	============================ test session starts ============================
	platform win32 -- Python 3.7.7, pytest-5.4.1, py-1.8.1, pluggy-0.13.1
	rootdir: d:\stack\PYTHON\REPOSITORIES\distfit
	collected 1 item

	tests\test_distfit.py .                                  [100%]

	============================ warnings summary ===============================
	tests/test_distfit.py::test_distfit
	tests/test_distfit.py::test_distfit
	tests/test_distfit.py::test_distfit
	tests/test_distfit.py::test_distfit
	tests/test_distfit.py::test_distfit
	tests/test_distfit.py::test_distfit

	====================== 1 passed, 8 warnings in 15.59s =======================




.. include:: add_bottom.add