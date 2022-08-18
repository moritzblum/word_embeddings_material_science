from distutils.core import setup

install_requires = [
    'numpy >= 1.21.4',
    'matplotlib >= 3.5.0',
    'scikit-learn',
    'nltk',
    'chemdataextractor',
    'gensim >= 4.1.2'
]

setup(
    python_requires='>=3.8',
    install_requires=install_requires,
)