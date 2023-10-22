from setuptools import setup, find_packages

setup(
    name='motion',
    version='0.1.0',
    packages=find_packages(include=['motion', 'motion.*']),

    #entry_points={
    #    'console_scripts': ['mp2bvh=mp2bvh.mp2bvh:main']
    #}
)


