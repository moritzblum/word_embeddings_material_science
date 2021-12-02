import os
import shutil

source = './data/BibDesk'

if os.path.exists(os.path.join(source, 'articles.bib')):
    os.remove(os.path.join(source, 'articles.bib'))

for path, _, files in os.walk(source):
    for name in files:
        shutil.copyfile(os.path.join(path, name), os.path.join('./data/pdf', path.split('/')[-1] + name))




