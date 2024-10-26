"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: A SHORT DESCRIPTION OF THE PROJECT                                                         -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: YOUR GITHUB USER NAME                                                                       -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: YOUR REPOSITORY URL                                                                     -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import numpy as np
import pandas as pd
from functions import DQR

# Cargar el dataset desde un archivo CSV
raw_data = pd.read_csv('Data/train-2.csv', low_memory=False)

# Limpiar los datos usando la clase DQR
DQR_instance = DQR(raw_data)
clean_data = DQR_instance.perform_clean()

# Guardar el dataset limpio en un archivo CSV
clean_data.to_csv('Data/clean_data.csv', index=False)

print("Dataset limpio guardado como 'Data/clean_data.csv'.")
