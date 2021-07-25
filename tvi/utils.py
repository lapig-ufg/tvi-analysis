import pandas as pd
import requests
import io
import gzip
import multiprocessing
from joblib import Parallel, delayed

ZENODO_SAMPLES = 'https://zenodo.org/record/5063025/files/brazil_lulc_samples_1985_2018_row_wise.csv.gz'

def ttprint(*args, **kwargs):
  from datetime import datetime
  import sys

  print(f'[{datetime.now():%H:%M:%S}] ', end='')
  print(*args, **kwargs, flush=True)

def read_samples(url):
    
  response = requests.get(url, stream=True)
  ttprint(f"Read samples from {url}")
  
  bytes_io = io.BytesIO(response.content)
  with gzip.open(bytes_io, 'rt') as read_file:
      samples = pd.read_csv(read_file)
      return samples

def save_csv(data, csv_file):
  csv_file.parent.mkdir(exist_ok=True)
  data.to_csv(csv_file)

def do_parallel(fn, args, backend='multiprocessing', n_jobs=multiprocessing.cpu_count()):
  return Parallel(n_jobs=n_jobs, backend=backend)(delayed(fn)(*arg) for arg in args)
