echo "Cleaning previous builds first.."
rm -rf dist
rm -rf build
rm -rf distfit.egg-info

echo "Making new build.."
echo ""

python setup.py bdist_wheel
echo ""

python setup.py sdist
echo ""

pip install -U dist/distfit-0.1.4-py3-none-any.whl
echo ""

read -p ">twine upload dist/* TO UPLOAD TO PYPI..."
echo ""
