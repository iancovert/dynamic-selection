import setuptools

setuptools.setup(
    name='dynamic-selection',
    version='0.0.1',
    author='Ian Covert',
    author_email='icovert@cs.washington.edu',
    description='Greedy dynamic feature selection.',
    long_description='''
        Greedy dynamic feature selection based on conditional mutual information.
    ''',
    long_description_content_type='text/markdown',
    url='',
    packages=['dynamic_selection'],
    install_requires=[
        'numpy',
        'torch>=1.13.1',
        'pandas>=1.5.2',
        'torchmetrics>=0.11.0',
        'pytorch_lightning>=1.8.6'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.7',
)
