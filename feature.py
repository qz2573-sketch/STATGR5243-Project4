#!/usr/bin/env python 3.11
# -*- coding: utf-8 -*-
# time: 2026/04/29
# name: Haowen Cui

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score