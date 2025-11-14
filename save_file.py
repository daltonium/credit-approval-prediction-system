import urllib.request
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
urllib.request.urlretrieve(url, 'credit_approval_data.csv')
