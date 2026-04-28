from setuptools import find_packages, setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name='emotion_recognition',
    version="0.0.1",
    description="identifying emotions through facial expression in real time",
    author="Nicolas Marechal",
    install_requires=requirements,
    packages=find_packages(),
    test_suite="tests",
    include_package_data=True,
    zip_safe=False
)
