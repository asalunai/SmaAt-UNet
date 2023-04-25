#!/bin/bash

# Desinstala a porra toda
sudo apt-get -qq -y purge --auto-remove python3-pip
sudo apt-get -qq -y purge --auto-remove python-pip
sudo apt-get -qq -y purge --auto-remove python3.9
sudo apt-get -qq -y purge --auto-remove python3.8
sudo apt-get -qq -y purge --auto-remove python3.7

# Upgrade para tapar os buracos
sudo apt -qq -y update
sudo apt -qq -y upgrade

# Reinstala o que precisa
sudo apt-get -qq -y install python3.7
sudo apt-get -qq -y install python3.7-distutils
sudo apt-get -qq -y --reinstall --fix-missing install python3-pip 

# Acerta os alias do python só pra garantir
echo "alias python='/usr/bin/python3.7'" >> ~/.bashrc
echo "alias python3='/usr/bin/python3.7'" >> ~/.bashrc
. ~/.bashrc

# Se der merda no pip, acerta o arquivo:

cat >/usr/local/bin/pip <<EOL
#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import re
import sys
from pip._internal.cli.main import main
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
EOL

cat /usr/local/bin/pip

# Se depois disso, não funcionar, confere o alias do pip
#echo "alias pip='/usr/local/bin/pip'" >> ~/.bashrc
#echo "alias pip3='/usr/local/bin/pip'" >> ~/.bashrc
#. ~/.bashrc

# E, finalmente, instala os requisitos do algoritmo.
pip install -r requirements.txt
