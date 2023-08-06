import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import scipy.stats
import yfinance as yf

from sklearn.utils import check_random_state

from hmmlearn import hmm
import matplotlib

bvb = yf.Ticker('BVB.DE')

print(bvb.info)
bvb_data = bvb.history(period="max")
print(bvb_data)

bvb_df = yf.download('BVB.DE')
print(bvb_df)

plt.plot(bvb_df['Adj Close'])
plt.show()

# addeed tested comment




gen_model = hmm.GaussianHMM(n_components=4, covariance_type="full")
