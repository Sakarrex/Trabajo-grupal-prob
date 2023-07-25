import pandas as pd
import scipy.stats as st 
import matplotlib.pyplot as plt
import math 
import seaborn as sb
import os
import numpy as np



__dirname = os.path.dirname(__file__) + '/'

# cdf: cumulative distribution function (area under the curve) (number -> probability)
# ppf: percent point function (inverse of cdf — percentiles) (probability -> number)
# pdf: probability density function (not cumulative)

alpha = 0.05

key1 = 'altura'
key2 = 'peso'
dataframe = pd.read_csv(__dirname + 'PesoAltura.csv') [[key1, key2]] # .head(200) 