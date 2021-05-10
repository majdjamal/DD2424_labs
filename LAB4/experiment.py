
__author__ = 'Majd Jamal'

import numpy as np
from data.getData import getData
from utils.utils import Params
from model import VRNN

data = getData()
params = Params(
    m = 100, seq_length = 25, eta = 0.1, sig = 0.1
    )

vrnn = VRNN()
vrnn.fit(data, params)
