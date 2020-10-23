# Gradient Boosting Classifier on DARKNET 2020: A Project Report


```python
#importing python libraries
import numpy as np
import pandas as pd
```


```python
#Reading the dataset: Darknet 2020
df = pd.read_csv('darknet.csv', error_bad_lines=False)
df.columns = df.columns.str.strip()
df.head(10)
```

    b'Skipping line 328: expected 85 fields, saw 125\n'
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Flow ID</th>
      <th>Src IP</th>
      <th>Src Port</th>
      <th>Dst IP</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Timestamp</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>...</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
      <th>Label</th>
      <th>Label.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.152.152.11-216.58.220.99-57158-443-6</td>
      <td>10.152.152.11</td>
      <td>57158</td>
      <td>216.58.220.99</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>229</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.152.152.11-216.58.220.99-57159-443-6</td>
      <td>10.152.152.11</td>
      <td>57159</td>
      <td>216.58.220.99</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>407</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.152.152.11-216.58.220.99-57160-443-6</td>
      <td>10.152.152.11</td>
      <td>57160</td>
      <td>216.58.220.99</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>431</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.152.152.11-74.125.136.120-49134-443-6</td>
      <td>10.152.152.11</td>
      <td>49134</td>
      <td>74.125.136.120</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>359</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.152.152.11-173.194.65.127-34697-19305-6</td>
      <td>10.152.152.11</td>
      <td>34697</td>
      <td>173.194.65.127</td>
      <td>19305</td>
      <td>6</td>
      <td>24/07/2015 04:09:45 PM</td>
      <td>10778451</td>
      <td>591</td>
      <td>400</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.437765e+15</td>
      <td>3.117718e+06</td>
      <td>1.437765e+15</td>
      <td>1.437765e+15</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10.152.152.11-173.194.65.127-54570-443-6</td>
      <td>10.152.152.11</td>
      <td>54570</td>
      <td>173.194.65.127</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:10:00 PM</td>
      <td>421362</td>
      <td>5</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.437765e+15</td>
      <td>1.866111e+05</td>
      <td>1.437765e+15</td>
      <td>1.437765e+15</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>6</th>
      <td>173.194.33.97-10.152.152.11-443-56254-6</td>
      <td>173.194.33.97</td>
      <td>443</td>
      <td>10.152.152.11</td>
      <td>56254</td>
      <td>6</td>
      <td>24/07/2015 04:09:45 PM</td>
      <td>119682119</td>
      <td>488</td>
      <td>487</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.437765e+15</td>
      <td>3.184630e+07</td>
      <td>1.437765e+15</td>
      <td>1.437765e+15</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10.152.152.11-216.58.216.142-57361-443-6</td>
      <td>10.152.152.11</td>
      <td>57361</td>
      <td>216.58.216.142</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:46 PM</td>
      <td>116996934</td>
      <td>369</td>
      <td>378</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.437765e+15</td>
      <td>3.617028e+07</td>
      <td>1.437765e+15</td>
      <td>1.437765e+15</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>8</th>
      <td>74.125.28.189-10.152.152.11-443-44097-6</td>
      <td>74.125.28.189</td>
      <td>443</td>
      <td>10.152.152.11</td>
      <td>44097</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>100279453</td>
      <td>61</td>
      <td>60</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.437765e+15</td>
      <td>3.327790e+07</td>
      <td>1.437765e+15</td>
      <td>1.437765e+15</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10.152.152.11-173.194.65.127-34702-19305-6</td>
      <td>10.152.152.11</td>
      <td>34702</td>
      <td>173.194.65.127</td>
      <td>19305</td>
      <td>6</td>
      <td>24/07/2015 04:10:00 PM</td>
      <td>119962833</td>
      <td>3638</td>
      <td>3157</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.437765e+15</td>
      <td>3.298254e+07</td>
      <td>1.437765e+15</td>
      <td>1.437765e+15</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 85 columns</p>
</div>




```python
#A look at all the 85 columns of the dataset
df.columns
```




    Index(['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol',
           'Timestamp', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
           'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
           'Fwd Packet Length Max', 'Fwd Packet Length Min',
           'Fwd Packet Length Mean', 'Fwd Packet Length Std',
           'Bwd Packet Length Max', 'Bwd Packet Length Min',
           'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
           'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
           'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
           'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
           'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
           'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
           'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
           'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
           'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
           'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
           'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
           'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
           'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
           'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
           'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
           'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes',
           'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
           'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
           'Idle Min', 'Label', 'Label.1'],
          dtype='object')




```python
#Getting all the columns with total number of entries in each column and data type
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 141530 entries, 0 to 141529
    Data columns (total 85 columns):
     #   Column                      Non-Null Count   Dtype  
    ---  ------                      --------------   -----  
     0   Flow ID                     141530 non-null  object 
     1   Src IP                      141530 non-null  object 
     2   Src Port                    141530 non-null  int64  
     3   Dst IP                      141530 non-null  object 
     4   Dst Port                    141530 non-null  int64  
     5   Protocol                    141530 non-null  int64  
     6   Timestamp                   141530 non-null  object 
     7   Flow Duration               141530 non-null  int64  
     8   Total Fwd Packet            141530 non-null  int64  
     9   Total Bwd packets           141530 non-null  int64  
     10  Total Length of Fwd Packet  141530 non-null  int64  
     11  Total Length of Bwd Packet  141530 non-null  int64  
     12  Fwd Packet Length Max       141530 non-null  int64  
     13  Fwd Packet Length Min       141530 non-null  int64  
     14  Fwd Packet Length Mean      141530 non-null  float64
     15  Fwd Packet Length Std       141530 non-null  float64
     16  Bwd Packet Length Max       141530 non-null  int64  
     17  Bwd Packet Length Min       141530 non-null  int64  
     18  Bwd Packet Length Mean      141530 non-null  float64
     19  Bwd Packet Length Std       141530 non-null  float64
     20  Flow Bytes/s                141483 non-null  float64
     21  Flow Packets/s              141530 non-null  float64
     22  Flow IAT Mean               141530 non-null  float64
     23  Flow IAT Std                141530 non-null  float64
     24  Flow IAT Max                141530 non-null  int64  
     25  Flow IAT Min                141530 non-null  int64  
     26  Fwd IAT Total               141530 non-null  int64  
     27  Fwd IAT Mean                141530 non-null  float64
     28  Fwd IAT Std                 141530 non-null  float64
     29  Fwd IAT Max                 141530 non-null  int64  
     30  Fwd IAT Min                 141530 non-null  int64  
     31  Bwd IAT Total               141530 non-null  int64  
     32  Bwd IAT Mean                141530 non-null  float64
     33  Bwd IAT Std                 141530 non-null  float64
     34  Bwd IAT Max                 141530 non-null  int64  
     35  Bwd IAT Min                 141530 non-null  int64  
     36  Fwd PSH Flags               141530 non-null  int64  
     37  Bwd PSH Flags               141530 non-null  int64  
     38  Fwd URG Flags               141530 non-null  int64  
     39  Bwd URG Flags               141530 non-null  int64  
     40  Fwd Header Length           141530 non-null  int64  
     41  Bwd Header Length           141530 non-null  int64  
     42  Fwd Packets/s               141530 non-null  float64
     43  Bwd Packets/s               141530 non-null  float64
     44  Packet Length Min           141530 non-null  int64  
     45  Packet Length Max           141530 non-null  int64  
     46  Packet Length Mean          141530 non-null  float64
     47  Packet Length Std           141530 non-null  float64
     48  Packet Length Variance      141530 non-null  float64
     49  FIN Flag Count              141530 non-null  int64  
     50  SYN Flag Count              141530 non-null  int64  
     51  RST Flag Count              141530 non-null  int64  
     52  PSH Flag Count              141530 non-null  int64  
     53  ACK Flag Count              141530 non-null  int64  
     54  URG Flag Count              141530 non-null  int64  
     55  CWE Flag Count              141530 non-null  int64  
     56  ECE Flag Count              141530 non-null  int64  
     57  Down/Up Ratio               141530 non-null  int64  
     58  Average Packet Size         141530 non-null  float64
     59  Fwd Segment Size Avg        141530 non-null  float64
     60  Bwd Segment Size Avg        141530 non-null  float64
     61  Fwd Bytes/Bulk Avg          141530 non-null  int64  
     62  Fwd Packet/Bulk Avg         141530 non-null  int64  
     63  Fwd Bulk Rate Avg           141530 non-null  int64  
     64  Bwd Bytes/Bulk Avg          141530 non-null  int64  
     65  Bwd Packet/Bulk Avg         141530 non-null  int64  
     66  Bwd Bulk Rate Avg           141530 non-null  int64  
     67  Subflow Fwd Packets         141530 non-null  int64  
     68  Subflow Fwd Bytes           141530 non-null  int64  
     69  Subflow Bwd Packets         141530 non-null  int64  
     70  Subflow Bwd Bytes           141530 non-null  int64  
     71  FWD Init Win Bytes          141530 non-null  int64  
     72  Bwd Init Win Bytes          141530 non-null  int64  
     73  Fwd Act Data Pkts           141530 non-null  int64  
     74  Fwd Seg Size Min            141530 non-null  int64  
     75  Active Mean                 141530 non-null  int64  
     76  Active Std                  141530 non-null  int64  
     77  Active Max                  141530 non-null  int64  
     78  Active Min                  141530 non-null  int64  
     79  Idle Mean                   141530 non-null  float64
     80  Idle Std                    141530 non-null  float64
     81  Idle Max                    141530 non-null  float64
     82  Idle Min                    141530 non-null  float64
     83  Label                       141530 non-null  object 
     84  Label.1                     141530 non-null  object 
    dtypes: float64(24), int64(55), object(6)
    memory usage: 91.8+ MB
    


```python
#Correlation matrix to know the dependency of columns on each other
corr = df.corr()
corr.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Src Port</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>Total Length of Fwd Packet</th>
      <th>Total Length of Bwd Packet</th>
      <th>Fwd Packet Length Max</th>
      <th>Fwd Packet Length Min</th>
      <th>...</th>
      <th>Fwd Act Data Pkts</th>
      <th>Fwd Seg Size Min</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Src Port</th>
      <td>1.000000</td>
      <td>-0.246451</td>
      <td>-0.097384</td>
      <td>0.065328</td>
      <td>-0.036259</td>
      <td>-0.014248</td>
      <td>-0.019712</td>
      <td>-0.008271</td>
      <td>0.076486</td>
      <td>-0.090022</td>
      <td>...</td>
      <td>-0.028619</td>
      <td>0.138354</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.073311</td>
      <td>0.058631</td>
      <td>0.077870</td>
      <td>0.031325</td>
    </tr>
    <tr>
      <th>Dst Port</th>
      <td>-0.246451</td>
      <td>1.000000</td>
      <td>-0.321199</td>
      <td>0.039227</td>
      <td>0.022094</td>
      <td>0.014775</td>
      <td>0.004451</td>
      <td>0.010865</td>
      <td>0.004448</td>
      <td>-0.178715</td>
      <td>...</td>
      <td>0.014722</td>
      <td>0.246275</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.069794</td>
      <td>-0.000706</td>
      <td>0.064385</td>
      <td>0.060121</td>
    </tr>
    <tr>
      <th>Protocol</th>
      <td>-0.097384</td>
      <td>-0.321199</td>
      <td>1.000000</td>
      <td>-0.266954</td>
      <td>-0.034735</td>
      <td>-0.026164</td>
      <td>-0.023039</td>
      <td>-0.020874</td>
      <td>-0.195123</td>
      <td>0.564044</td>
      <td>...</td>
      <td>-0.023370</td>
      <td>-0.872467</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.222855</td>
      <td>-0.155843</td>
      <td>-0.236588</td>
      <td>-0.129431</td>
    </tr>
    <tr>
      <th>Flow Duration</th>
      <td>0.065328</td>
      <td>0.039227</td>
      <td>-0.266954</td>
      <td>1.000000</td>
      <td>0.142110</td>
      <td>0.100288</td>
      <td>0.072529</td>
      <td>0.057008</td>
      <td>0.340744</td>
      <td>-0.068930</td>
      <td>...</td>
      <td>0.145455</td>
      <td>0.240411</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.505336</td>
      <td>0.127573</td>
      <td>0.506074</td>
      <td>0.409007</td>
    </tr>
    <tr>
      <th>Total Fwd Packet</th>
      <td>-0.036259</td>
      <td>0.022094</td>
      <td>-0.034735</td>
      <td>0.142110</td>
      <td>1.000000</td>
      <td>0.744834</td>
      <td>0.457391</td>
      <td>0.635688</td>
      <td>0.125575</td>
      <td>-0.020982</td>
      <td>...</td>
      <td>0.698507</td>
      <td>0.029652</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.066400</td>
      <td>-0.017736</td>
      <td>0.062264</td>
      <td>0.074038</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 79 columns</p>
</div>




```python
df.dropna() #dropping null values
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Flow ID</th>
      <th>Src IP</th>
      <th>Src Port</th>
      <th>Dst IP</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Timestamp</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>...</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
      <th>Label</th>
      <th>Label.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.152.152.11-216.58.220.99-57158-443-6</td>
      <td>10.152.152.11</td>
      <td>57158</td>
      <td>216.58.220.99</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>229</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.152.152.11-216.58.220.99-57159-443-6</td>
      <td>10.152.152.11</td>
      <td>57159</td>
      <td>216.58.220.99</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>407</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.152.152.11-216.58.220.99-57160-443-6</td>
      <td>10.152.152.11</td>
      <td>57160</td>
      <td>216.58.220.99</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>431</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.152.152.11-74.125.136.120-49134-443-6</td>
      <td>10.152.152.11</td>
      <td>49134</td>
      <td>74.125.136.120</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>359</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.152.152.11-173.194.65.127-34697-19305-6</td>
      <td>10.152.152.11</td>
      <td>34697</td>
      <td>173.194.65.127</td>
      <td>19305</td>
      <td>6</td>
      <td>24/07/2015 04:09:45 PM</td>
      <td>10778451</td>
      <td>591</td>
      <td>400</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.437765e+15</td>
      <td>3.117718e+06</td>
      <td>1.437765e+15</td>
      <td>1.437765e+15</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>141525</th>
      <td>10.8.8.246-224.0.0.252-55219-5355-17</td>
      <td>10.8.8.246</td>
      <td>55219</td>
      <td>224.0.0.252</td>
      <td>5355</td>
      <td>17</td>
      <td>22/05/2015 01:55:03 PM</td>
      <td>411806</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>VPN</td>
      <td>VOIP</td>
    </tr>
    <tr>
      <th>141526</th>
      <td>10.8.8.246-224.0.0.252-64207-5355-17</td>
      <td>10.8.8.246</td>
      <td>64207</td>
      <td>224.0.0.252</td>
      <td>5355</td>
      <td>17</td>
      <td>22/05/2015 02:09:05 PM</td>
      <td>411574</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>VPN</td>
      <td>VOIP</td>
    </tr>
    <tr>
      <th>141527</th>
      <td>10.8.8.246-224.0.0.252-61115-5355-17</td>
      <td>10.8.8.246</td>
      <td>61115</td>
      <td>224.0.0.252</td>
      <td>5355</td>
      <td>17</td>
      <td>22/05/2015 02:19:31 PM</td>
      <td>422299</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>VPN</td>
      <td>VOIP</td>
    </tr>
    <tr>
      <th>141528</th>
      <td>10.8.8.246-224.0.0.252-64790-5355-17</td>
      <td>10.8.8.246</td>
      <td>64790</td>
      <td>224.0.0.252</td>
      <td>5355</td>
      <td>17</td>
      <td>22/05/2015 02:29:55 PM</td>
      <td>411855</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>VPN</td>
      <td>VOIP</td>
    </tr>
    <tr>
      <th>141529</th>
      <td>80.239.235.110-10.8.8.246-11666-60245-17</td>
      <td>80.239.235.110</td>
      <td>11666</td>
      <td>10.8.8.246</td>
      <td>60245</td>
      <td>17</td>
      <td>22/05/2015 02:31:23 PM</td>
      <td>119990044</td>
      <td>5995</td>
      <td>6000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.432316e+15</td>
      <td>3.463689e+07</td>
      <td>1.432316e+15</td>
      <td>1.432316e+15</td>
      <td>VPN</td>
      <td>VOIP</td>
    </tr>
  </tbody>
</table>
<p>141483 rows × 85 columns</p>
</div>




```python
df.isnull().sum().head() #to check the number of null values in each column
```




    Flow ID     0
    Src IP      0
    Src Port    0
    Dst IP      0
    Dst Port    0
    dtype: int64




```python
df['Label'].value_counts()  #to check the number of classes in Label
```




    Non-Tor    93356
    NonVPN     23863
    VPN        22919
    Tor         1392
    Name: Label, dtype: int64




```python
df['Label.1'].value_counts() #to check the number of classes in Label
```




    P2P                48520
    Browsing           32808
    Audio-Streaming    16580
    Chat               11478
    File-Transfer      11098
    Video-Streaming     9486
    Email               6145
    VOIP                3566
    AUDIO-STREAMING     1484
    Video-streaming      281
    File-transfer         84
    Name: Label.1, dtype: int64




```python

```


```python
#splitting the Src IP into octets,getting first two ocets
newIP = []
for value in df['Src IP']:
    IP = value.split(".")
    octet1= IP[0]
    octet2= IP[1]
#     print(octet2)
    newIP.append(float(octet1 + '.' + octet2))

    
```


```python
df1 = pd.DataFrame(newIP)  #a new dataframe with the above obtained series
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.152</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.152</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.152</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.152</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.152</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Src IP'] = df1  #replacing column Src IP with df1
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Flow ID</th>
      <th>Src IP</th>
      <th>Src Port</th>
      <th>Dst IP</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Timestamp</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>...</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
      <th>Label</th>
      <th>Label.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.152.152.11-216.58.220.99-57158-443-6</td>
      <td>10.152</td>
      <td>57158</td>
      <td>216.58.220.99</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>229</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.152.152.11-216.58.220.99-57159-443-6</td>
      <td>10.152</td>
      <td>57159</td>
      <td>216.58.220.99</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>407</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.152.152.11-216.58.220.99-57160-443-6</td>
      <td>10.152</td>
      <td>57160</td>
      <td>216.58.220.99</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>431</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.152.152.11-74.125.136.120-49134-443-6</td>
      <td>10.152</td>
      <td>49134</td>
      <td>74.125.136.120</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>359</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.152.152.11-173.194.65.127-34697-19305-6</td>
      <td>10.152</td>
      <td>34697</td>
      <td>173.194.65.127</td>
      <td>19305</td>
      <td>6</td>
      <td>24/07/2015 04:09:45 PM</td>
      <td>10778451</td>
      <td>591</td>
      <td>400</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.437765e+15</td>
      <td>3.117718e+06</td>
      <td>1.437765e+15</td>
      <td>1.437765e+15</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 85 columns</p>
</div>




```python
newIP1 = [] #splitting the Dst IP into octets,getting first two ocets
for value in df['Dst IP']:
    IP = value.split(".")
    octet1= IP[0]
    octet2= IP[1]
    
#     print(octet2)
    newIP1.append(float(octet1 + '.' + octet2))
```


```python
df2 = pd.DataFrame(newIP1)
df2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>216.580</td>
    </tr>
    <tr>
      <th>1</th>
      <td>216.580</td>
    </tr>
    <tr>
      <th>2</th>
      <td>216.580</td>
    </tr>
    <tr>
      <th>3</th>
      <td>74.125</td>
    </tr>
    <tr>
      <th>4</th>
      <td>173.194</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Dst IP'] = df2
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Flow ID</th>
      <th>Src IP</th>
      <th>Src Port</th>
      <th>Dst IP</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Timestamp</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>...</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
      <th>Label</th>
      <th>Label.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.152.152.11-216.58.220.99-57158-443-6</td>
      <td>10.152</td>
      <td>57158</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>229</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.152.152.11-216.58.220.99-57159-443-6</td>
      <td>10.152</td>
      <td>57159</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>407</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.152.152.11-216.58.220.99-57160-443-6</td>
      <td>10.152</td>
      <td>57160</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>431</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.152.152.11-74.125.136.120-49134-443-6</td>
      <td>10.152</td>
      <td>49134</td>
      <td>74.125</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>359</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.152.152.11-173.194.65.127-34697-19305-6</td>
      <td>10.152</td>
      <td>34697</td>
      <td>173.194</td>
      <td>19305</td>
      <td>6</td>
      <td>24/07/2015 04:09:45 PM</td>
      <td>10778451</td>
      <td>591</td>
      <td>400</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.437765e+15</td>
      <td>3.117718e+06</td>
      <td>1.437765e+15</td>
      <td>1.437765e+15</td>
      <td>Non-Tor</td>
      <td>AUDIO-STREAMING</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 85 columns</p>
</div>




```python
# label encoding the data : Label and Label.1
from sklearn.preprocessing import LabelEncoder 
  
Le = LabelEncoder() 
  
df['Label']= Le.fit_transform(df['Label'])
df['Label.1']= Le.fit_transform(df['Label.1'])

```


```python
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Flow ID</th>
      <th>Src IP</th>
      <th>Src Port</th>
      <th>Dst IP</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Timestamp</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>...</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
      <th>Label</th>
      <th>Label.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.152.152.11-216.58.220.99-57158-443-6</td>
      <td>10.152</td>
      <td>57158</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>229</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.152.152.11-216.58.220.99-57159-443-6</td>
      <td>10.152</td>
      <td>57159</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>407</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.152.152.11-216.58.220.99-57160-443-6</td>
      <td>10.152</td>
      <td>57160</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>431</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.152.152.11-74.125.136.120-49134-443-6</td>
      <td>10.152</td>
      <td>49134</td>
      <td>74.125</td>
      <td>443</td>
      <td>6</td>
      <td>24/07/2015 04:09:48 PM</td>
      <td>359</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.152.152.11-173.194.65.127-34697-19305-6</td>
      <td>10.152</td>
      <td>34697</td>
      <td>173.194</td>
      <td>19305</td>
      <td>6</td>
      <td>24/07/2015 04:09:45 PM</td>
      <td>10778451</td>
      <td>591</td>
      <td>400</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.437765e+15</td>
      <td>3.117718e+06</td>
      <td>1.437765e+15</td>
      <td>1.437765e+15</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 85 columns</p>
</div>




```python
df5=df.drop(['Flow ID','Timestamp'], axis = 1) #dropping the unnecessary columns
df5.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Src IP</th>
      <th>Src Port</th>
      <th>Dst IP</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>Total Length of Fwd Packet</th>
      <th>Total Length of Bwd Packet</th>
      <th>...</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
      <th>Label</th>
      <th>Label.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.152</td>
      <td>57158</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>229</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.152</td>
      <td>57159</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>407</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.152</td>
      <td>57160</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>431</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.152</td>
      <td>49134</td>
      <td>74.125</td>
      <td>443</td>
      <td>6</td>
      <td>359</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.152</td>
      <td>34697</td>
      <td>173.194</td>
      <td>19305</td>
      <td>6</td>
      <td>10778451</td>
      <td>591</td>
      <td>400</td>
      <td>64530</td>
      <td>6659</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.437765e+15</td>
      <td>3.117718e+06</td>
      <td>1.437765e+15</td>
      <td>1.437765e+15</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 83 columns</p>
</div>





In the next 4 steps, i will be performing some operations on the below columns to convert it from exponential values to normal float values.




```python
df5['Idle Mean']=df5['Idle Mean']/1e15
```


```python
df5['Idle Max']=df5['Idle Max']/1e15
```


```python
df5['Idle Min']=df5['Idle Min']/1e15
```


```python
df5['Idle Std']=df5['Idle Std']/1e7
```


```python
df5.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Src IP</th>
      <th>Src Port</th>
      <th>Dst IP</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>Total Length of Fwd Packet</th>
      <th>Total Length of Bwd Packet</th>
      <th>...</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
      <th>Label</th>
      <th>Label.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.152</td>
      <td>57158</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>229</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.152</td>
      <td>57159</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>407</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.152</td>
      <td>57160</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>431</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.152</td>
      <td>49134</td>
      <td>74.125</td>
      <td>443</td>
      <td>6</td>
      <td>359</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.152</td>
      <td>34697</td>
      <td>173.194</td>
      <td>19305</td>
      <td>6</td>
      <td>10778451</td>
      <td>591</td>
      <td>400</td>
      <td>64530</td>
      <td>6659</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.437765</td>
      <td>0.311772</td>
      <td>1.437765</td>
      <td>1.437765</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 83 columns</p>
</div>




```python
df5.isnull().values.any() 
```




    True




```python
df5.fillna(df5.mean()).head(5) #filling null values with the mean of the column
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Src IP</th>
      <th>Src Port</th>
      <th>Dst IP</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>Total Length of Fwd Packet</th>
      <th>Total Length of Bwd Packet</th>
      <th>...</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
      <th>Label</th>
      <th>Label.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.152</td>
      <td>57158</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>229</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.152</td>
      <td>57159</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>407</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.152</td>
      <td>57160</td>
      <td>216.580</td>
      <td>443</td>
      <td>6</td>
      <td>431</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.152</td>
      <td>49134</td>
      <td>74.125</td>
      <td>443</td>
      <td>6</td>
      <td>359</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.152</td>
      <td>34697</td>
      <td>173.194</td>
      <td>19305</td>
      <td>6</td>
      <td>10778451</td>
      <td>591</td>
      <td>400</td>
      <td>64530</td>
      <td>6659</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.437765</td>
      <td>0.311772</td>
      <td>1.437765</td>
      <td>1.437765</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 83 columns</p>
</div>




```python
df5.notnull().values.all() #this shows there are no more null values


```




    False




```python
df5=df5.astype(float)
```


```python
# Replacing infinite with nan 
df5.replace([np.inf, -np.inf], np.nan, inplace=True) 
  
# Dropping all the rows with nan values 
df5.dropna(inplace=True) 
```


```python
df5.head(5)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Src IP</th>
      <th>Src Port</th>
      <th>Dst IP</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>Total Length of Fwd Packet</th>
      <th>Total Length of Bwd Packet</th>
      <th>...</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
      <th>Label</th>
      <th>Label.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.152</td>
      <td>57158.0</td>
      <td>216.580</td>
      <td>443.0</td>
      <td>6.0</td>
      <td>229.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.152</td>
      <td>57159.0</td>
      <td>216.580</td>
      <td>443.0</td>
      <td>6.0</td>
      <td>407.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.152</td>
      <td>57160.0</td>
      <td>216.580</td>
      <td>443.0</td>
      <td>6.0</td>
      <td>431.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.152</td>
      <td>49134.0</td>
      <td>74.125</td>
      <td>443.0</td>
      <td>6.0</td>
      <td>359.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.152</td>
      <td>34697.0</td>
      <td>173.194</td>
      <td>19305.0</td>
      <td>6.0</td>
      <td>10778451.0</td>
      <td>591.0</td>
      <td>400.0</td>
      <td>64530.0</td>
      <td>6659.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.437765</td>
      <td>0.311772</td>
      <td>1.437765</td>
      <td>1.437765</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 83 columns</p>
</div>




```python
df5['Label'].value_counts()
```




    0.0    93309
    1.0    23861
    3.0    22919
    2.0     1392
    Name: Label, dtype: int64



So the data is highly imbalanced.....


Now I will undersample class 0.0 wrt class 1.0 and
oversample class 3.0 wrt class 2.0



```python
#1. Find the number of the minority class
non_tor = len(df5[df5['Label']==0])
non_vpn = len(df5[df5['Label']==1])
vpn = len(df5[df5['Label']==2])
tor = len(df5[df5['Label']==3])

print(non_tor)
print(non_vpn)
print(vpn)
print(tor)
```

    93309
    23861
    1392
    22919
    


```python
index_non_tor = df5[df5['Label']==0].index
index_non_vpn = df5[df5['Label']==1].index
index_tor = df5[df5['Label']==2].index
index_vpn = df5[df5['Label']==3].index
```


```python
#4. Randomly sample the majority indices with respect to the number of minority classes
random_indices = np.random.choice(index_non_tor,non_vpn,replace='False')
```


```python
#5. Concat the minority indices with the indices from step 4
under_sample_indices = np.concatenate([index_non_vpn,random_indices])
```


```python
#Get the balanced dataframe - This is the final undersampled data
under_sample_df = df5.iloc[under_sample_indices]
under_sample_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Src IP</th>
      <th>Src Port</th>
      <th>Dst IP</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>Total Length of Fwd Packet</th>
      <th>Total Length of Bwd Packet</th>
      <th>...</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
      <th>Label</th>
      <th>Label.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>93403</th>
      <td>131.202</td>
      <td>64717.0</td>
      <td>131.202</td>
      <td>13000.0</td>
      <td>6.0</td>
      <td>81.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>93404</th>
      <td>131.202</td>
      <td>42530.0</td>
      <td>178.237</td>
      <td>443.0</td>
      <td>6.0</td>
      <td>119829241.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.225994</td>
      <td>5.406124e+07</td>
      <td>1.430326</td>
      <td>2.982862e-08</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>93405</th>
      <td>131.202</td>
      <td>42534.0</td>
      <td>178.237</td>
      <td>443.0</td>
      <td>6.0</td>
      <td>119828205.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.225994</td>
      <td>5.406124e+07</td>
      <td>1.430326</td>
      <td>2.982767e-08</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>93406</th>
      <td>131.202</td>
      <td>17208.0</td>
      <td>77.720</td>
      <td>11113.0</td>
      <td>17.0</td>
      <td>138272.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>126.0</td>
      <td>85.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.430326</td>
      <td>9.762882e-03</td>
      <td>1.430326</td>
      <td>1.430326e+00</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>93407</th>
      <td>8.600</td>
      <td>0.0</td>
      <td>8.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5103.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 83 columns</p>
</div>




```python
under_sample_df['Label'].value_counts()
```




    0.0    23846
    1.0    23827
    2.0       49
    Name: Label, dtype: int64




```python
vpn_sample = df5[df5['Label']==2].sample(tor, replace=True)
```


```python
#create a new dataframe containing only tor data
df_tor = df5[df5['Label']==3]
```


```python
over_sample_df = pd.concat([vpn_sample,df_tor], axis=0)
```


```python
over_sample_class_counts=pd.value_counts(over_sample_df['Label'])
```


```python
over_sample_class_counts
```




    3.0    22919
    2.0    22919
    Name: Label, dtype: int64




```python
balance_df = pd.concat([under_sample_df,over_sample_df], axis=0)

```


```python
balance_df['Label'].value_counts()
```




    0.0    23846
    1.0    23827
    2.0    22968
    3.0    22919
    Name: Label, dtype: int64




```python
balance_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Src IP</th>
      <th>Src Port</th>
      <th>Dst IP</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>Total Length of Fwd Packet</th>
      <th>Total Length of Bwd Packet</th>
      <th>...</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
      <th>Label</th>
      <th>Label.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>93403</th>
      <td>131.202</td>
      <td>64717.0</td>
      <td>131.202</td>
      <td>13000.0</td>
      <td>6.0</td>
      <td>81.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>93404</th>
      <td>131.202</td>
      <td>42530.0</td>
      <td>178.237</td>
      <td>443.0</td>
      <td>6.0</td>
      <td>119829241.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.225994</td>
      <td>5.406124e+07</td>
      <td>1.430326</td>
      <td>2.982862e-08</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>93405</th>
      <td>131.202</td>
      <td>42534.0</td>
      <td>178.237</td>
      <td>443.0</td>
      <td>6.0</td>
      <td>119828205.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.225994</td>
      <td>5.406124e+07</td>
      <td>1.430326</td>
      <td>2.982767e-08</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>93406</th>
      <td>131.202</td>
      <td>17208.0</td>
      <td>77.720</td>
      <td>11113.0</td>
      <td>17.0</td>
      <td>138272.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>126.0</td>
      <td>85.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.430326</td>
      <td>9.762882e-03</td>
      <td>1.430326</td>
      <td>1.430326e+00</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>93407</th>
      <td>8.600</td>
      <td>0.0</td>
      <td>8.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5103.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 83 columns</p>
</div>




```python
#Forming a new dataframe for the target variable and removing it from the above dataset.
target = balance_df.filter(['Label'], axis=1)
target.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>93403</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>93404</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>93405</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>93406</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>93407</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
balance_df = balance_df.drop('Label', 1)
balance_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Src IP</th>
      <th>Src Port</th>
      <th>Dst IP</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>Total Length of Fwd Packet</th>
      <th>Total Length of Bwd Packet</th>
      <th>...</th>
      <th>Fwd Seg Size Min</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
      <th>Label.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>93403</th>
      <td>131.202</td>
      <td>64717.0</td>
      <td>131.202</td>
      <td>13000.0</td>
      <td>6.0</td>
      <td>81.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>93404</th>
      <td>131.202</td>
      <td>42530.0</td>
      <td>178.237</td>
      <td>443.0</td>
      <td>6.0</td>
      <td>119829241.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.225994</td>
      <td>5.406124e+07</td>
      <td>1.430326</td>
      <td>2.982862e-08</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>93405</th>
      <td>131.202</td>
      <td>42534.0</td>
      <td>178.237</td>
      <td>443.0</td>
      <td>6.0</td>
      <td>119828205.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.225994</td>
      <td>5.406124e+07</td>
      <td>1.430326</td>
      <td>2.982767e-08</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>93406</th>
      <td>131.202</td>
      <td>17208.0</td>
      <td>77.720</td>
      <td>11113.0</td>
      <td>17.0</td>
      <td>138272.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>126.0</td>
      <td>85.0</td>
      <td>...</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.430326</td>
      <td>9.762882e-03</td>
      <td>1.430326</td>
      <td>1.430326e+00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>93407</th>
      <td>8.600</td>
      <td>0.0</td>
      <td>8.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5103.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>




```python
#MinMaxScaling

import pandas as pd
from sklearn import preprocessing

x = balance_df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_final = pd.DataFrame(x_scaled,columns = balance_df.columns)
df_final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Src IP</th>
      <th>Src Port</th>
      <th>Dst IP</th>
      <th>Dst Port</th>
      <th>Protocol</th>
      <th>Flow Duration</th>
      <th>Total Fwd Packet</th>
      <th>Total Bwd packets</th>
      <th>Total Length of Fwd Packet</th>
      <th>Total Length of Bwd Packet</th>
      <th>...</th>
      <th>Fwd Seg Size Min</th>
      <th>Active Mean</th>
      <th>Active Std</th>
      <th>Active Max</th>
      <th>Active Min</th>
      <th>Idle Mean</th>
      <th>Idle Std</th>
      <th>Idle Max</th>
      <th>Idle Min</th>
      <th>Label.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.587586</td>
      <td>0.987533</td>
      <td>0.512169</td>
      <td>0.198367</td>
      <td>0.352941</td>
      <td>6.666667e-07</td>
      <td>0.000004</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.454545</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.587586</td>
      <td>0.648976</td>
      <td>0.697131</td>
      <td>0.006760</td>
      <td>0.352941</td>
      <td>9.985770e-01</td>
      <td>0.000017</td>
      <td>0.000006</td>
      <td>3.119689e-08</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.454545</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.841788</td>
      <td>5.249503e-01</td>
      <td>0.982086</td>
      <td>2.048083e-08</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.587586</td>
      <td>0.649037</td>
      <td>0.697131</td>
      <td>0.006760</td>
      <td>0.352941</td>
      <td>9.985684e-01</td>
      <td>0.000017</td>
      <td>0.000006</td>
      <td>3.119689e-08</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.454545</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.841788</td>
      <td>5.249503e-01</td>
      <td>0.982086</td>
      <td>2.048017e-08</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.587586</td>
      <td>0.262581</td>
      <td>0.301854</td>
      <td>0.169574</td>
      <td>1.000000</td>
      <td>1.152258e-03</td>
      <td>0.000004</td>
      <td>0.000004</td>
      <td>1.637837e-07</td>
      <td>1.267845e-07</td>
      <td>...</td>
      <td>0.181818</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.982086</td>
      <td>9.480039e-11</td>
      <td>0.982086</td>
      <td>9.820856e-01</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.038515</td>
      <td>0.000000</td>
      <td>0.027684</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.251667e-05</td>
      <td>0.000004</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.3</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>




```python
#DOING THE TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
# split into train test sets
y=target
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

    (65492, 82) (28068, 82) (65492, 1) (28068, 1)
    


```python
from numpy import array
from sklearn.model_selection import KFold
# data sample
data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# prepare cross validation
kfold = KFold(3, True, 1)
# enumerate splits
for train, test in kfold.split(data):
	print('train: %s, test: %s' % (X_train, X_test))
```

    train: [[1.98520000e+02 4.43000000e+02 1.00000000e+01 ... 1.43776581e+00
      1.43776569e+00 8.00000000e+00]
     [1.31202000e+02 5.29660000e+04 2.24000000e+02 ... 0.00000000e+00
      0.00000000e+00 4.00000000e+00]
     [1.31202000e+02 5.11390000e+04 2.24000000e+02 ... 0.00000000e+00
      0.00000000e+00 4.00000000e+00]
     ...
     [1.08000000e+01 4.78180000e+04 1.31202000e+02 ... 0.00000000e+00
      0.00000000e+00 1.00000000e+00]
     [1.31202000e+02 6.48850000e+04 2.24000000e+02 ... 0.00000000e+00
      0.00000000e+00 8.00000000e+00]
     [1.08000000e+01 5.99820000e+04 1.98700000e+02 ... 0.00000000e+00
      0.00000000e+00 3.00000000e+00]], test: [[1.01520000e+01 3.83740000e+04 1.85310000e+02 ... 1.45625355e+00
      1.45625353e+00 2.00000000e+00]
     [1.01520000e+01 5.86820000e+04 5.41920000e+01 ... 1.45640750e+00
      5.14311000e-09 2.00000000e+00]
     [1.31202000e+02 4.91130000e+04 1.29330000e+02 ... 0.00000000e+00
      0.00000000e+00 9.00000000e+00]
     ...
     [9.61600000e+01 4.43000000e+02 1.08000000e+01 ... 0.00000000e+00
      0.00000000e+00 9.00000000e+00]
     [1.29550000e+02 0.00000000e+00 9.60000000e-01 ... 1.43257411e+00
      5.99801440e-08 3.00000000e+00]
     [1.31202000e+02 6.18000000e+03 1.31202000e+02 ... 0.00000000e+00
      0.00000000e+00 8.00000000e+00]]
    train: [[1.98520000e+02 4.43000000e+02 1.00000000e+01 ... 1.43776581e+00
      1.43776569e+00 8.00000000e+00]
     [1.31202000e+02 5.29660000e+04 2.24000000e+02 ... 0.00000000e+00
      0.00000000e+00 4.00000000e+00]
     [1.31202000e+02 5.11390000e+04 2.24000000e+02 ... 0.00000000e+00
      0.00000000e+00 4.00000000e+00]
     ...
     [1.08000000e+01 4.78180000e+04 1.31202000e+02 ... 0.00000000e+00
      0.00000000e+00 1.00000000e+00]
     [1.31202000e+02 6.48850000e+04 2.24000000e+02 ... 0.00000000e+00
      0.00000000e+00 8.00000000e+00]
     [1.08000000e+01 5.99820000e+04 1.98700000e+02 ... 0.00000000e+00
      0.00000000e+00 3.00000000e+00]], test: [[1.01520000e+01 3.83740000e+04 1.85310000e+02 ... 1.45625355e+00
      1.45625353e+00 2.00000000e+00]
     [1.01520000e+01 5.86820000e+04 5.41920000e+01 ... 1.45640750e+00
      5.14311000e-09 2.00000000e+00]
     [1.31202000e+02 4.91130000e+04 1.29330000e+02 ... 0.00000000e+00
      0.00000000e+00 9.00000000e+00]
     ...
     [9.61600000e+01 4.43000000e+02 1.08000000e+01 ... 0.00000000e+00
      0.00000000e+00 9.00000000e+00]
     [1.29550000e+02 0.00000000e+00 9.60000000e-01 ... 1.43257411e+00
      5.99801440e-08 3.00000000e+00]
     [1.31202000e+02 6.18000000e+03 1.31202000e+02 ... 0.00000000e+00
      0.00000000e+00 8.00000000e+00]]
    train: [[1.98520000e+02 4.43000000e+02 1.00000000e+01 ... 1.43776581e+00
      1.43776569e+00 8.00000000e+00]
     [1.31202000e+02 5.29660000e+04 2.24000000e+02 ... 0.00000000e+00
      0.00000000e+00 4.00000000e+00]
     [1.31202000e+02 5.11390000e+04 2.24000000e+02 ... 0.00000000e+00
      0.00000000e+00 4.00000000e+00]
     ...
     [1.08000000e+01 4.78180000e+04 1.31202000e+02 ... 0.00000000e+00
      0.00000000e+00 1.00000000e+00]
     [1.31202000e+02 6.48850000e+04 2.24000000e+02 ... 0.00000000e+00
      0.00000000e+00 8.00000000e+00]
     [1.08000000e+01 5.99820000e+04 1.98700000e+02 ... 0.00000000e+00
      0.00000000e+00 3.00000000e+00]], test: [[1.01520000e+01 3.83740000e+04 1.85310000e+02 ... 1.45625355e+00
      1.45625353e+00 2.00000000e+00]
     [1.01520000e+01 5.86820000e+04 5.41920000e+01 ... 1.45640750e+00
      5.14311000e-09 2.00000000e+00]
     [1.31202000e+02 4.91130000e+04 1.29330000e+02 ... 0.00000000e+00
      0.00000000e+00 9.00000000e+00]
     ...
     [9.61600000e+01 4.43000000e+02 1.08000000e+01 ... 0.00000000e+00
      0.00000000e+00 9.00000000e+00]
     [1.29550000e+02 0.00000000e+00 9.60000000e-01 ... 1.43257411e+00
      5.99801440e-08 3.00000000e+00]
     [1.31202000e+02 6.18000000e+03 1.31202000e+02 ... 0.00000000e+00
      0.00000000e+00 8.00000000e+00]]
    


```python
#FIRST I AM RUNNING WITHOUT HYPERPARAMETERE TUNING|
```


```python
# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
```


```python
# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)
```

    C:\Users\kshitij\anaconda3\lib\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    


```python
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

    Accuracy: 0.9261792788941143
    


```python
#NOW I NEED TO DO HYPER PARAMATER TUNING OF ADABOOST
# example of grid searching key hyperparameters for adaboost on a classification dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
# define dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=6)
# define the model with default hyperparameters
model = AdaBoostClassifier()
# define the grid of values to search
grid = dict()
grid['n_estimators'] = [10, 50, 100]
grid['learning_rate'] = [ 0.01, 0.1, 1.0]
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
# execute the grid search
grid_result = grid_search.fit(X_train, y_train)
# summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# summarize all scores that were evaluated
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
```

    C:\Users\kshitij\anaconda3\lib\site-packages\sklearn\utils\validation.py:73: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      return f(**kwargs)
    

    Best: 0.940079 using {'learning_rate': 0.1, 'n_estimators': 100}
    0.780778 (0.106851) with: {'learning_rate': 0.01, 'n_estimators': 10}
    0.896613 (0.003173) with: {'learning_rate': 0.01, 'n_estimators': 50}
    0.904090 (0.002843) with: {'learning_rate': 0.01, 'n_estimators': 100}
    0.853213 (0.026589) with: {'learning_rate': 0.1, 'n_estimators': 10}
    0.922662 (0.003224) with: {'learning_rate': 0.1, 'n_estimators': 50}
    0.940079 (0.004076) with: {'learning_rate': 0.1, 'n_estimators': 100}
    0.930164 (0.004291) with: {'learning_rate': 1.0, 'n_estimators': 10}
    0.922357 (0.018836) with: {'learning_rate': 1.0, 'n_estimators': 50}
    0.917186 (0.014116) with: {'learning_rate': 1.0, 'n_estimators': 100}
    


```python

```
