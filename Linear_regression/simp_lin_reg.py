# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 11:34:14 2017

@author: rajor
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('F:\data for ML\lin reg/data.csv')
df=pd.DataFrame(df)
X=df.x
Y=df.y
df.head()
xm=X.mean()
ym=Y.mean()
c1=(np.sum((X-xm)*(Y-ym)))/(np.sum((X-xm)**2))
c0=ym-c1*xm
r=np.sum((X-xm)*(Y-ym))/(np.sum(((X-xm)**2))*np.sum(((Y-ym)**2)))**(0.5)
yh=c0+c1*X
fig, ax = plt.subplots()
fit = np.polyfit(X, Y, deg=1)
ax.plot(X, yh, color='yellow')
ax.scatter(X, Y)
fig.show()









