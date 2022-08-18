from distutils.core import setup
import urllib.request

install_requires = [
    'numpy >= 1.21.4',
    'matplotlib >= 3.5.0',
    'scikit-learn',
    'nltk',
    'chemdataextractor',
    'gensim >= 4.1.2'
]

# download table about chemical elements
with urllib.request.urlopen('https://gist.githubusercontent.com/GoodmanSciences/c2dd862cd38f21b0ad36b8f96b4bf1ee/raw/1d92663004489a5b6926e944c1b3d9ec5c40900e/Periodic%2520Table%2520of%2520Elements.csv') as f:
    element_data = f.read().decode('utf-8')

with open('./data/elements.csv','w') as output:
    for line in element_data:
        output.write(line)


setup(
    python_requires='>=3.8',
    install_requires=install_requires,
)