# importing libraries


```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
```


# importing dataset


```python
data = pd.read_csv("C:/Users/mypc/Downloads/creditcard.csv")
```


```python
print(data.shape) 
print(data.describe()) 
```

    (284807, 31)
                    Time            V1            V2            V3            V4  \
    count  284807.000000  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean    94813.859575  3.919560e-15  5.688174e-16 -8.769071e-15  2.782312e-15   
    std     47488.145955  1.958696e+00  1.651309e+00  1.516255e+00  1.415869e+00   
    min         0.000000 -5.640751e+01 -7.271573e+01 -4.832559e+01 -5.683171e+00   
    25%     54201.500000 -9.203734e-01 -5.985499e-01 -8.903648e-01 -8.486401e-01   
    50%     84692.000000  1.810880e-02  6.548556e-02  1.798463e-01 -1.984653e-02   
    75%    139320.500000  1.315642e+00  8.037239e-01  1.027196e+00  7.433413e-01   
    max    172792.000000  2.454930e+00  2.205773e+01  9.382558e+00  1.687534e+01   
    
                     V5            V6            V7            V8            V9  \
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean  -1.552563e-15  2.010663e-15 -1.694249e-15 -1.927028e-16 -3.137024e-15   
    std    1.380247e+00  1.332271e+00  1.237094e+00  1.194353e+00  1.098632e+00   
    min   -1.137433e+02 -2.616051e+01 -4.355724e+01 -7.321672e+01 -1.343407e+01   
    25%   -6.915971e-01 -7.682956e-01 -5.540759e-01 -2.086297e-01 -6.430976e-01   
    50%   -5.433583e-02 -2.741871e-01  4.010308e-02  2.235804e-02 -5.142873e-02   
    75%    6.119264e-01  3.985649e-01  5.704361e-01  3.273459e-01  5.971390e-01   
    max    3.480167e+01  7.330163e+01  1.205895e+02  2.000721e+01  1.559499e+01   
    
           ...           V21           V22           V23           V24  \
    count  ...  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05   
    mean   ...  1.537294e-16  7.959909e-16  5.367590e-16  4.458112e-15   
    std    ...  7.345240e-01  7.257016e-01  6.244603e-01  6.056471e-01   
    min    ... -3.483038e+01 -1.093314e+01 -4.480774e+01 -2.836627e+00   
    25%    ... -2.283949e-01 -5.423504e-01 -1.618463e-01 -3.545861e-01   
    50%    ... -2.945017e-02  6.781943e-03 -1.119293e-02  4.097606e-02   
    75%    ...  1.863772e-01  5.285536e-01  1.476421e-01  4.395266e-01   
    max    ...  2.720284e+01  1.050309e+01  2.252841e+01  4.584549e+00   
    
                    V25           V26           V27           V28         Amount  \
    count  2.848070e+05  2.848070e+05  2.848070e+05  2.848070e+05  284807.000000   
    mean   1.453003e-15  1.699104e-15 -3.660161e-16 -1.206049e-16      88.349619   
    std    5.212781e-01  4.822270e-01  4.036325e-01  3.300833e-01     250.120109   
    min   -1.029540e+01 -2.604551e+00 -2.256568e+01 -1.543008e+01       0.000000   
    25%   -3.171451e-01 -3.269839e-01 -7.083953e-02 -5.295979e-02       5.600000   
    50%    1.659350e-02 -5.213911e-02  1.342146e-03  1.124383e-02      22.000000   
    75%    3.507156e-01  2.409522e-01  9.104512e-02  7.827995e-02      77.165000   
    max    7.519589e+00  3.517346e+00  3.161220e+01  3.384781e+01   25691.160000   
    
                   Class  
    count  284807.000000  
    mean        0.001727  
    std         0.041527  
    min         0.000000  
    25%         0.000000  
    50%         0.000000  
    75%         0.000000  
    max         1.000000  
    
    [8 rows x 31 columns]
    


```python
fraud = data[data['Class'] == 1] 
valid = data[data['Class'] == 0] 
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1]))) 
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))
```

    Fraud Cases: 492
    Valid Transactions: 284315
    


```python
fraud.describe()
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>...</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.000000</td>
      <td>492.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>80746.806911</td>
      <td>-4.771948</td>
      <td>3.623778</td>
      <td>-7.033281</td>
      <td>4.542029</td>
      <td>-3.151225</td>
      <td>-1.397737</td>
      <td>-5.568731</td>
      <td>0.570636</td>
      <td>-2.581123</td>
      <td>...</td>
      <td>0.713588</td>
      <td>0.014049</td>
      <td>-0.040308</td>
      <td>-0.105130</td>
      <td>0.041449</td>
      <td>0.051648</td>
      <td>0.170575</td>
      <td>0.075667</td>
      <td>122.211321</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>47835.365138</td>
      <td>6.783687</td>
      <td>4.291216</td>
      <td>7.110937</td>
      <td>2.873318</td>
      <td>5.372468</td>
      <td>1.858124</td>
      <td>7.206773</td>
      <td>6.797831</td>
      <td>2.500896</td>
      <td>...</td>
      <td>3.869304</td>
      <td>1.494602</td>
      <td>1.579642</td>
      <td>0.515577</td>
      <td>0.797205</td>
      <td>0.471679</td>
      <td>1.376766</td>
      <td>0.547291</td>
      <td>256.683288</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>406.000000</td>
      <td>-30.552380</td>
      <td>-8.402154</td>
      <td>-31.103685</td>
      <td>-1.313275</td>
      <td>-22.105532</td>
      <td>-6.406267</td>
      <td>-43.557242</td>
      <td>-41.044261</td>
      <td>-13.434066</td>
      <td>...</td>
      <td>-22.797604</td>
      <td>-8.887017</td>
      <td>-19.254328</td>
      <td>-2.028024</td>
      <td>-4.781606</td>
      <td>-1.152671</td>
      <td>-7.263482</td>
      <td>-1.869290</td>
      <td>0.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>41241.500000</td>
      <td>-6.036063</td>
      <td>1.188226</td>
      <td>-8.643489</td>
      <td>2.373050</td>
      <td>-4.792835</td>
      <td>-2.501511</td>
      <td>-7.965295</td>
      <td>-0.195336</td>
      <td>-3.872383</td>
      <td>...</td>
      <td>0.041787</td>
      <td>-0.533764</td>
      <td>-0.342175</td>
      <td>-0.436809</td>
      <td>-0.314348</td>
      <td>-0.259416</td>
      <td>-0.020025</td>
      <td>-0.108868</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>75568.500000</td>
      <td>-2.342497</td>
      <td>2.717869</td>
      <td>-5.075257</td>
      <td>4.177147</td>
      <td>-1.522962</td>
      <td>-1.424616</td>
      <td>-3.034402</td>
      <td>0.621508</td>
      <td>-2.208768</td>
      <td>...</td>
      <td>0.592146</td>
      <td>0.048434</td>
      <td>-0.073135</td>
      <td>-0.060795</td>
      <td>0.088371</td>
      <td>0.004321</td>
      <td>0.394926</td>
      <td>0.146344</td>
      <td>9.250000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>128483.000000</td>
      <td>-0.419200</td>
      <td>4.971257</td>
      <td>-2.276185</td>
      <td>6.348729</td>
      <td>0.214562</td>
      <td>-0.413216</td>
      <td>-0.945954</td>
      <td>1.764879</td>
      <td>-0.787850</td>
      <td>...</td>
      <td>1.244611</td>
      <td>0.617474</td>
      <td>0.308378</td>
      <td>0.285328</td>
      <td>0.456515</td>
      <td>0.396733</td>
      <td>0.826029</td>
      <td>0.381152</td>
      <td>105.890000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>170348.000000</td>
      <td>2.132386</td>
      <td>22.057729</td>
      <td>2.250210</td>
      <td>12.114672</td>
      <td>11.095089</td>
      <td>6.474115</td>
      <td>5.802537</td>
      <td>20.007208</td>
      <td>3.353525</td>
      <td>...</td>
      <td>27.202839</td>
      <td>8.361985</td>
      <td>5.466230</td>
      <td>1.091435</td>
      <td>2.208209</td>
      <td>2.745261</td>
      <td>3.052358</td>
      <td>1.779364</td>
      <td>2125.870000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 31 columns</p>
</div>




```python
fraud.Amount.describe()
```




    count     492.000000
    mean      122.211321
    std       256.683288
    min         0.000000
    25%         1.000000
    50%         9.250000
    75%       105.890000
    max      2125.870000
    Name: Amount, dtype: float64




```python
data[data['Class'] == 1]
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
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>541</th>
      <td>406.0</td>
      <td>-2.312227</td>
      <td>1.951992</td>
      <td>-1.609851</td>
      <td>3.997906</td>
      <td>-0.522188</td>
      <td>-1.426545</td>
      <td>-2.537387</td>
      <td>1.391657</td>
      <td>-2.770089</td>
      <td>...</td>
      <td>0.517232</td>
      <td>-0.035049</td>
      <td>-0.465211</td>
      <td>0.320198</td>
      <td>0.044519</td>
      <td>0.177840</td>
      <td>0.261145</td>
      <td>-0.143276</td>
      <td>0.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>623</th>
      <td>472.0</td>
      <td>-3.043541</td>
      <td>-3.157307</td>
      <td>1.088463</td>
      <td>2.288644</td>
      <td>1.359805</td>
      <td>-1.064823</td>
      <td>0.325574</td>
      <td>-0.067794</td>
      <td>-0.270953</td>
      <td>...</td>
      <td>0.661696</td>
      <td>0.435477</td>
      <td>1.375966</td>
      <td>-0.293803</td>
      <td>0.279798</td>
      <td>-0.145362</td>
      <td>-0.252773</td>
      <td>0.035764</td>
      <td>529.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4920</th>
      <td>4462.0</td>
      <td>-2.303350</td>
      <td>1.759247</td>
      <td>-0.359745</td>
      <td>2.330243</td>
      <td>-0.821628</td>
      <td>-0.075788</td>
      <td>0.562320</td>
      <td>-0.399147</td>
      <td>-0.238253</td>
      <td>...</td>
      <td>-0.294166</td>
      <td>-0.932391</td>
      <td>0.172726</td>
      <td>-0.087330</td>
      <td>-0.156114</td>
      <td>-0.542628</td>
      <td>0.039566</td>
      <td>-0.153029</td>
      <td>239.93</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6108</th>
      <td>6986.0</td>
      <td>-4.397974</td>
      <td>1.358367</td>
      <td>-2.592844</td>
      <td>2.679787</td>
      <td>-1.128131</td>
      <td>-1.706536</td>
      <td>-3.496197</td>
      <td>-0.248778</td>
      <td>-0.247768</td>
      <td>...</td>
      <td>0.573574</td>
      <td>0.176968</td>
      <td>-0.436207</td>
      <td>-0.053502</td>
      <td>0.252405</td>
      <td>-0.657488</td>
      <td>-0.827136</td>
      <td>0.849573</td>
      <td>59.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6329</th>
      <td>7519.0</td>
      <td>1.234235</td>
      <td>3.019740</td>
      <td>-4.304597</td>
      <td>4.732795</td>
      <td>3.624201</td>
      <td>-1.357746</td>
      <td>1.713445</td>
      <td>-0.496358</td>
      <td>-1.282858</td>
      <td>...</td>
      <td>-0.379068</td>
      <td>-0.704181</td>
      <td>-0.656805</td>
      <td>-1.632653</td>
      <td>1.488901</td>
      <td>0.566797</td>
      <td>-0.010016</td>
      <td>0.146793</td>
      <td>1.00</td>
      <td>1</td>
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
      <th>279863</th>
      <td>169142.0</td>
      <td>-1.927883</td>
      <td>1.125653</td>
      <td>-4.518331</td>
      <td>1.749293</td>
      <td>-1.566487</td>
      <td>-2.010494</td>
      <td>-0.882850</td>
      <td>0.697211</td>
      <td>-2.064945</td>
      <td>...</td>
      <td>0.778584</td>
      <td>-0.319189</td>
      <td>0.639419</td>
      <td>-0.294885</td>
      <td>0.537503</td>
      <td>0.788395</td>
      <td>0.292680</td>
      <td>0.147968</td>
      <td>390.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>280143</th>
      <td>169347.0</td>
      <td>1.378559</td>
      <td>1.289381</td>
      <td>-5.004247</td>
      <td>1.411850</td>
      <td>0.442581</td>
      <td>-1.326536</td>
      <td>-1.413170</td>
      <td>0.248525</td>
      <td>-1.127396</td>
      <td>...</td>
      <td>0.370612</td>
      <td>0.028234</td>
      <td>-0.145640</td>
      <td>-0.081049</td>
      <td>0.521875</td>
      <td>0.739467</td>
      <td>0.389152</td>
      <td>0.186637</td>
      <td>0.76</td>
      <td>1</td>
    </tr>
    <tr>
      <th>280149</th>
      <td>169351.0</td>
      <td>-0.676143</td>
      <td>1.126366</td>
      <td>-2.213700</td>
      <td>0.468308</td>
      <td>-1.120541</td>
      <td>-0.003346</td>
      <td>-2.234739</td>
      <td>1.210158</td>
      <td>-0.652250</td>
      <td>...</td>
      <td>0.751826</td>
      <td>0.834108</td>
      <td>0.190944</td>
      <td>0.032070</td>
      <td>-0.739695</td>
      <td>0.471111</td>
      <td>0.385107</td>
      <td>0.194361</td>
      <td>77.89</td>
      <td>1</td>
    </tr>
    <tr>
      <th>281144</th>
      <td>169966.0</td>
      <td>-3.113832</td>
      <td>0.585864</td>
      <td>-5.399730</td>
      <td>1.817092</td>
      <td>-0.840618</td>
      <td>-2.943548</td>
      <td>-2.208002</td>
      <td>1.058733</td>
      <td>-1.632333</td>
      <td>...</td>
      <td>0.583276</td>
      <td>-0.269209</td>
      <td>-0.456108</td>
      <td>-0.183659</td>
      <td>-0.328168</td>
      <td>0.606116</td>
      <td>0.884876</td>
      <td>-0.253700</td>
      <td>245.00</td>
      <td>1</td>
    </tr>
    <tr>
      <th>281674</th>
      <td>170348.0</td>
      <td>1.991976</td>
      <td>0.158476</td>
      <td>-2.583441</td>
      <td>0.408670</td>
      <td>1.151147</td>
      <td>-0.096695</td>
      <td>0.223050</td>
      <td>-0.068384</td>
      <td>0.577829</td>
      <td>...</td>
      <td>-0.164350</td>
      <td>-0.295135</td>
      <td>-0.072173</td>
      <td>-0.450261</td>
      <td>0.313267</td>
      <td>-0.289617</td>
      <td>0.002988</td>
      <td>-0.015309</td>
      <td>42.53</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>492 rows × 31 columns</p>
</div>




```python
X = data.drop(['Class'], axis = 1) 
Y = data["Class"] 
print(X.shape) 
print(Y.shape)
```

    (284807, 30)
    (284807,)
    


```python
corrmat = data.corr() 
fig = plt.figure(figsize = (12, 9)) 
sns.heatmap(corrmat, vmax = .8, square = True) 
plt.show() 
```


    
![png](output_10_0.png)
    



```python
sns.set(style="darkgrid")
ax = sns.countplot(x='Class', data=data)
```


    
![png](output_11_0.png)
    



```python
xData = X.values 
yData = Y.values 
```

# splitting


```python
from sklearn.model_selection import train_test_split 
xTrain, xTest, yTrain, yTest = train_test_split(xData, yData, test_size = 0.2, random_state = 42) 
```

# scaling


```python
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xTrain=sc.fit_transform(xTrain)
xTest=sc.transform(xTest)
```

# randomforest


```python
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier() 
rfc.fit(xTrain, yTrain) 
yPred = rfc.predict(xTest) 
```


```python
from sklearn.metrics import accuracy_score 
acc = accuracy_score(yTest, yPred) 
print("The accuracy is {}".format(acc)) 
```

    The accuracy is 0.9995962220427653
    


```python
from sklearn.metrics import confusion_matrix 
LABELS = ['Normal', 'Fraud'] 
conf_matrix = confusion_matrix(yTest, yPred) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 
```


    
![png](output_20_0.png)
    


# k neighbour


```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(xTrain, yTrain)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=1, p=2,
                         weights='uniform')




```python
Pred1 = knn.predict(xTest)
acc1 = accuracy_score(yTest, Pred1)
print("The accuracy using KneignborsClassifier is {}".format(acc1)) 
```

    The accuracy using KneignborsClassifier is 0.9982795547909132
    

# gradient boost


```python
from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier(random_state=0)
gbt.fit(xTrain, yTrain)
pred2=gbt.predict(xTest)
acc2 = accuracy_score(yTest, pred2)
```


```python
print("The accuracy using GradientBoostingClassifier is {}".format(acc2)) 
```

    The accuracy using GradientBoostingClassifier is 0.9989466661985184
    

# decision tree


```python
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=0)
dtc.fit(xTrain, yTrain)
pred3=dtc.predict(xTest)
acc3 = accuracy_score(yTest, pred3)
```


```python
print("The accuracy using DecisionTreeClassifier is {}".format(acc3)) 
```

    The accuracy using DecisionTreeClassifier is 0.9990695551420246
    

# naive bayes


```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
X, Y = load_iris(return_X_y=True)
gnb = GaussianNB()
gnb.fit(xTrain, yTrain)
pred4=gnb.predict(xTest)
acc4 = accuracy_score(yTest, pred4)
```


```python
print("The accuracy using Naive Bayes is {}".format(acc4)) 
```

    The accuracy using Naive Bayes is 0.9778273234788104
    

# logistic regression 


```python
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(xTrain, yTrain)
pred5=lr.predict(xTest)
acc5 = accuracy_score(yTest, pred5)

```


```python
print("The accuracy using LogisticRegression is {}".format(acc5)) 
```

    The accuracy using LogisticRegression is 0.9991222218320986
    

# xgboost


```python
from xgboost import XGBClassifier
xgb =XGBClassifier(random_state=0)
xgb.fit(xTrain, yTrain)
pred_6=xgb.predict(xTest)
acc_6 = accuracy_score(yTest, pred_6)
```


```python
print("The accuracy using xgboost is {}".format(acc_6))
```

    The accuracy using xgboost is 0.9995786664794073
    


```python

```
