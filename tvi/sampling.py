import pandas as pd
import numpy as np
import math

import warnings
import scipy.stats as st
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

from .utils import *

LULC_COLORS_DEFAULT = pd.DataFrame(
    columns=['lulc', 'color'],
    data=[
        ['Pasture','#ffc100'],
        ['Annual crop','#D5A6BD'],
        ['Tree plantation','#935132'],
        ['Semi-perennial crop','#C27BA0'],
        ['Urban infrastructure','#af2a2a'],
        ['Wetland','#18b08d'],
        ['Grassland formation','#B8AF4F'],
        ['Forest formation','#006400'],
        ['Savanna formation','#32CD32'],
        ['Water','#0000FF'],
        ['Other','#5f5f5f'],
        ['Perennial crop','#D5A6BD'],
        ['Other non-forest natural formation','#BDB76B']
    ]
)

class Area_Estimator:

  def __init__(self, samples, weight_col, strata_col, lulc_col, id_col,
    year_col, pixel_size, confidence_interval, lulc_colors=None, verbose = True):
      
    self.samples = samples
    self.weight_col = weight_col
    self.lulc_col = lulc_col
    self.id_col = id_col
    self.year_col = year_col
    self.pixel_size = pixel_size
    self.strata_col = strata_col
    self.verbose = verbose
    self.confidence_interval = confidence_interval

    # Density points according the confidence interval
    self.std_norm = round(st.norm.ppf(1 - ( 1 - confidence_interval ) / 2), 2)

    # Area in hectares
    self.pixel_area = (self.pixel_size * self.pixel_size) / 10000

    self.lulc_list = self._unique_vals(self.lulc_col)
    self.year_list = self._unique_vals(self.year_col)

    # Min and max years for the lulc change analysis
    self.min_year = self.samples[self.year_col].min()
    self.max_year = self.samples[self.year_col].max()

    if lulc_colors is None:
        self.lulc_colors = LULC_COLORS_DEFAULT
    else:
        self.lulc_colors = lulc_colors

  def _unique_vals(self, column):
    return list(self.samples[column].unique())

  def _verbose(self, *args, **kwargs):
    if self.verbose:
      ttprint(*args, **kwargs)

  def _population(self, samples):
    population = (1 * samples[self.weight_col]).sum()
    return population, (population * self.pixel_area)

  def _calc_se(self, samples, mask, population):

    def _strata_variance(samples_strata, var_map_correct_s):
      
      nsamples_s, _ = samples_strata.shape      
      population_s = (1 * samples_strata[self.weight_col]).sum()

      strata_var = 0
      if population_s > 0:
        strata_var = math.pow(population_s,2)  \
                * (1 - nsamples_s / population_s) \
                * var_map_correct_s / nsamples_s
      return strata_var

    args = []
    var_map_correct_s = np.var(mask.astype('int'))

    for name, samples_strata in samples.groupby(self.strata_col):
      args.append((samples_strata, var_map_correct_s))

    glob_var = 0
    for strata_var in do_parallel(_strata_variance, args, backend='threading'):
      glob_var += strata_var

    glob_var = 1 / math.pow(population, 2) * glob_var
    glob_se = self.std_norm * math.sqrt(glob_var)

    return glob_se

  def _calc_area(self, samples, year, value_col, value_list, region_label):
    
    result = []

    for value in value_list:

      lulc_mask = (samples[value_col] == value)
      
      samples.loc[:, 'ESTIMATOR'] = 0
      samples.loc[lulc_mask, 'ESTIMATOR'] = 1
      
      population, area_population = self._population(samples)

      lulc_proportion = ((samples['ESTIMATOR'] * samples[self.weight_col]).sum()) / population
      lulc_se = self._calc_se(samples, lulc_mask, population)
      
      lulc_area = lulc_proportion * area_population
      result.append([value, lulc_area, lulc_proportion, lulc_se, year, region_label])

    return result

  def _filter_samples(self, region_filter = None):
    return self.samples if (region_filter is None) else self.samples[region_filter]

  def _valid_year_range(self, year, n_years, backward = False):
    step = -1 if backward else 1    
    
    start_year = year + step
    end_year = year + (n_years * step) + 1

    end_year = self.min_year - 1 if end_year < self.min_year else end_year
    end_year = self.max_year + 1 if end_year > self.max_year else end_year

    #if start_year == end_year:
    #  return [ year ]
    #else:
    result =[ y for y in range(start_year, end_year, step) ]
    return result

  def _change_mask(self, samples, lulc_arr, year, past_arr, past_nyears, future_arr, future_nyears):
    
    past_years = self._valid_year_range(year, past_nyears, backward = True)
    future_years = self._valid_year_range(year, future_nyears, backward = False)
    
    # Considering all the samples
    past_mask = np.logical_and(
      self.samples[self.lulc_col].isin(past_arr),
      self.samples[self.year_col].isin(past_years)
    )
    #print(past_arr, past_years, np.unique(past_mask, return_counts=True))

    # Considering all the samples
    future_mask = np.logical_and(
      self.samples[self.lulc_col].isin(future_arr),
      self.samples[self.year_col].isin(future_years)
    )
    
    past_fur_mask = np.logical_or(past_mask, future_mask)
    n_years = len(past_years) + len(future_years)
    
    # Considering all the samples
    samples_ids = self.samples[[self.id_col]].copy()    

    samples_ids['past_fur_mask'] = 0
    samples_ids['past_fur_mask'].loc[past_fur_mask] = 1

    past_fur_agg = samples_ids[past_fur_mask][['id', 'past_fur_mask']].groupby('id').sum()
    past_fur_ids = past_fur_agg[past_fur_agg['past_fur_mask'] == n_years].index

    # Considering samples passed as params
    change_mask = np.logical_and(
        samples[self.lulc_col].isin(lulc_arr),
        samples[self.id_col].isin(past_fur_ids)
    )

    #print('change_mask', samples.shape)
    change_mask = np.logical_and(change_mask, samples[self.year_col] == year)
    #print(np.unique(change_mask, return_counts=True))

    return change_mask

  def lulc(self, lulc = None, year = None, region_label = 'Brazil', region_filter = None):
  
    args = []
    _samples = self._filter_samples(region_filter)
    _lulc_list = self.lulc_list if (lulc is None) else [lulc]
    _year_list = self.year_list if (year is None) else [year]
    
    result = []

    self._verbose(f'Estimating area of {len(_lulc_list)} LULC classes for {region_label} ({len(_year_list)} years)')
    for _year in _year_list:
      year_samples = _samples[_samples[self.year_col] == _year]
      args.append((year_samples, _year, self.lulc_col, _lulc_list, region_label))
    
    result = []
    for year_result in do_parallel(self._calc_area, args):
      result += year_result
    self._verbose(f'Finished')

    result = pd.DataFrame(result, columns=['lulc', 'area_ha', 'proportion', 'se', 'year', 'region'])
    result = result.merge(self.lulc_colors, on='lulc')

    return result

  def lulc_by_region(self, region_col, lulc = None, year = None):
    
    result = []

    for region in self._unique_vals(region_col):
      result.append(
        self.lulc(lulc=lulc, year=year, region_label=region, region_filter=(self.samples[region_col] == region))
      )
        
    return pd.concat(result)

  def lulc_change(self, lulc_change_label, lulc_arr, past_arr, past_nyears, future_arr, future_nyears, 
    start_year = None, end_year = None, cumsum = False, color = None, region_label = 'Brazil', region_filter = None):
    
    _samples = self._filter_samples(region_filter)

    start_year = self.year_list[0] if (start_year is None) else start_year
    end_year = self.year_list[len(self.year_list)] if (end_year is None) else end_year

    args = []
    self._verbose(f'Estimating lulc change area of {lulc_change_label} for {region_label} ({end_year - start_year} years)')
    for _year in range(start_year, end_year):

      year_samples = _samples[_samples[self.year_col] == _year]
      change_mask = self._change_mask(year_samples, lulc_arr, _year, past_arr, past_nyears, future_arr, future_nyears)
      
      change_col = 'lulc_change'
      year_samples[change_col] = 0
      year_samples[change_col].loc[change_mask] = 1

      args.append((year_samples, _year, change_col, [1], region_label))
    
    result = []
    for year_result in do_parallel(self._calc_area, args):
      result += year_result
    self._verbose(f'Finished')

    result = pd.DataFrame(result, columns=['lulc_change', 'area_ha', 'proportion', 'se', 'year', 'region'])
    result['lulc_change'] = lulc_change_label
    result['year'] += 1
    if color is not None:
      result['color'] = color  
    
    if cumsum:
      result['area_ha'] = result['area_ha'].cumsum()
      result['proportion'] = result['proportion'].cumsum()
      result['se'] = result['se'].cumsum()

    return result

  def lulc_change_by_region(self, region_col, lulc_change_label, lulc_arr, past_arr, past_nyears, future_arr, future_nyears, 
    start_year = None, end_year = None, color = None):
    
    result = []

    for region in self._unique_vals(region_col):
      result.append(
        self.lulc_change(lulc_change_label, lulc_arr, past_arr, past_nyears, future_arr, future_nyears, 
            start_year = start_year, end_year = end_year, region_label=region, 
            region_filter=(self.samples[region_col] == region), color = color)
      )
        
    return pd.concat(result)

  def stable_area(self, lulc = None, region_label = 'Brazil', region_filter = None, return_all_years=False):
  
    args = []
    _samples = self._filter_samples(region_filter)
    _lulc_list = self.lulc_list if (lulc is None) else [lulc]
  
    offset = math.floor((self.min_year - self.max_year)/2)
    mid_year = (self.max_year + offset)
    nyears = (self.max_year - self.min_year) + 1

    args = []
    self._verbose(f'Estimating stable area of {len(_lulc_list)} LULC classes for {region_label} ({nyears} years)')
    for _lulc in _lulc_list:
      args.append((
        _lulc,  #lulc_change_label
        [_lulc],           #lulc_arr
        [_lulc],           #past_arr
        nyears,           #past_nyears
        [_lulc],           #future_arr
        nyears,           #future_nyears
        mid_year,         #start_year
        mid_year + 1,     #end_year
        False,            #cumsum
        None,             #color
        region_label,     #region_label
        region_filter     #region_filter
      ))
    
    result = []
    for lulc_result in do_parallel(self.lulc_change, args, backend='threading'):
      result.append(lulc_result)
    self._verbose(f'Finished')

    result = pd.concat(result, axis=0).rename(columns={"lulc_change": "lulc"})
    result = result.merge(self.lulc_colors, on='lulc')

    if return_all_years:
      result_ts = []
      for year in range(self.min_year + 1, self.max_year + 1):
          result_y = result.copy()
          result_y['year'] = year
          result_ts.append(result_y)
      result = pd.concat(result_ts, axis=0)

    result['lulc'] = 'Stable ' + result['lulc']

    return result

  def plot(self, lulc_result, lulc_col = 'lulc', area_col = 'area_ha', year_col = 'year',
    se_col = 'se', color_col = 'color', area_scale = 1/1000000, area_label='Area (Mha)', 
    rc_params = None, title = None, stacked = False):
    
    import matplotlib
    import matplotlib.pyplot as plt

    if rc_params is None:
      rc_params = {
        'font.size': 15,
        'grid.color': '#cacaca'
      }

    matplotlib.rcParams.update(rc_params)

    fig, ax = plt.subplots(figsize=(10,7))
    plt.grid(True)
    plt.box(False)

    if title is not None:
      fig.suptitle(title, fontweight='bold')

    labels = []
    colors = []
    values = []

    for lulc in lulc_result[lulc_col].unique():
      data = lulc_result[lulc_result[lulc_col] == lulc]
      data[area_col] = data[area_col] * area_scale
      color = list(data[color_col].unique())[0]
      
      area_se = (data[area_col] * data[se_col])
      under_line = data[area_col] - area_se
      over_line = data[area_col] + area_se
      
      if stacked:
        labels.append(lulc)
        colors.append(color)
        values.append(data[area_col])
      else:
        ax.fill_between(data[year_col], under_line, over_line, color='gray', alpha=0.15)
        ax.plot(data[year_col], data[area_col],  marker='o', label=lulc, color=color, alpha=0.8, linewidth=1.5)

    if stacked:
      plt.stackplot(data[year_col], values, labels=labels, colors=colors)

    year_ticks = range(np.min(data[year_col]), np.max(data[year_col])+1, 3)

    plt.xlabel("Time")
    plt.ylabel(area_label)
    plt.xticks(year_ticks)
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.show()
