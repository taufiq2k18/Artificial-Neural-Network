.. code:: ipython3

    pwd




.. parsed-literal::

    'C:\\Users\\Taufiq Abdullah\\Untitled Folder\\Untitled Folder\\Untitled Folder\\Untitled Folder'



.. code:: ipython3

    import pandas as pd
    import numpy as np
    import matpotlib as mpl
    import matpotlib.pylot as plt
    


::


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-10-bf083920a6b9> in <module>
          1 import pandas as pd
          2 import numpy as np
    ----> 3 import matpotlib as mpl
          4 import matpotlib.pylot as plt
    

    ModuleNotFoundError: No module named 'matpotlib'


.. code:: ipython3

    dataset=pd.read_csv("train.csv")

.. code:: ipython3

    dataset




.. raw:: html

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
          <th>PassengerId</th>
          <th>Survived</th>
          <th>Pclass</th>
          <th>Name</th>
          <th>Sex</th>
          <th>Age</th>
          <th>SibSp</th>
          <th>Parch</th>
          <th>Ticket</th>
          <th>Fare</th>
          <th>Cabin</th>
          <th>Embarked</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>0</td>
          <td>3</td>
          <td>Braund, Mr. Owen Harris</td>
          <td>male</td>
          <td>22.0</td>
          <td>1</td>
          <td>0</td>
          <td>A/5 21171</td>
          <td>7.2500</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
          <td>female</td>
          <td>38.0</td>
          <td>1</td>
          <td>0</td>
          <td>PC 17599</td>
          <td>71.2833</td>
          <td>C85</td>
          <td>C</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>1</td>
          <td>3</td>
          <td>Heikkinen, Miss. Laina</td>
          <td>female</td>
          <td>26.0</td>
          <td>0</td>
          <td>0</td>
          <td>STON/O2. 3101282</td>
          <td>7.9250</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>1</td>
          <td>1</td>
          <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
          <td>female</td>
          <td>35.0</td>
          <td>1</td>
          <td>0</td>
          <td>113803</td>
          <td>53.1000</td>
          <td>C123</td>
          <td>S</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>0</td>
          <td>3</td>
          <td>Allen, Mr. William Henry</td>
          <td>male</td>
          <td>35.0</td>
          <td>0</td>
          <td>0</td>
          <td>373450</td>
          <td>8.0500</td>
          <td>NaN</td>
          <td>S</td>
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
        </tr>
        <tr>
          <th>886</th>
          <td>887</td>
          <td>0</td>
          <td>2</td>
          <td>Montvila, Rev. Juozas</td>
          <td>male</td>
          <td>27.0</td>
          <td>0</td>
          <td>0</td>
          <td>211536</td>
          <td>13.0000</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
        <tr>
          <th>887</th>
          <td>888</td>
          <td>1</td>
          <td>1</td>
          <td>Graham, Miss. Margaret Edith</td>
          <td>female</td>
          <td>19.0</td>
          <td>0</td>
          <td>0</td>
          <td>112053</td>
          <td>30.0000</td>
          <td>B42</td>
          <td>S</td>
        </tr>
        <tr>
          <th>888</th>
          <td>889</td>
          <td>0</td>
          <td>3</td>
          <td>Johnston, Miss. Catherine Helen "Carrie"</td>
          <td>female</td>
          <td>NaN</td>
          <td>1</td>
          <td>2</td>
          <td>W./C. 6607</td>
          <td>23.4500</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
        <tr>
          <th>889</th>
          <td>890</td>
          <td>1</td>
          <td>1</td>
          <td>Behr, Mr. Karl Howell</td>
          <td>male</td>
          <td>26.0</td>
          <td>0</td>
          <td>0</td>
          <td>111369</td>
          <td>30.0000</td>
          <td>C148</td>
          <td>C</td>
        </tr>
        <tr>
          <th>890</th>
          <td>891</td>
          <td>0</td>
          <td>3</td>
          <td>Dooley, Mr. Patrick</td>
          <td>male</td>
          <td>32.0</td>
          <td>0</td>
          <td>0</td>
          <td>370376</td>
          <td>7.7500</td>
          <td>NaN</td>
          <td>Q</td>
        </tr>
      </tbody>
    </table>
    <p>891 rows × 12 columns</p>
    </div>



.. code:: ipython3

    dataset.head(5)




.. raw:: html

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
          <th>PassengerId</th>
          <th>Survived</th>
          <th>Pclass</th>
          <th>Name</th>
          <th>Sex</th>
          <th>Age</th>
          <th>SibSp</th>
          <th>Parch</th>
          <th>Ticket</th>
          <th>Fare</th>
          <th>Cabin</th>
          <th>Embarked</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1</td>
          <td>0</td>
          <td>3</td>
          <td>Braund, Mr. Owen Harris</td>
          <td>male</td>
          <td>22.0</td>
          <td>1</td>
          <td>0</td>
          <td>A/5 21171</td>
          <td>7.2500</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2</td>
          <td>1</td>
          <td>1</td>
          <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
          <td>female</td>
          <td>38.0</td>
          <td>1</td>
          <td>0</td>
          <td>PC 17599</td>
          <td>71.2833</td>
          <td>C85</td>
          <td>C</td>
        </tr>
        <tr>
          <th>2</th>
          <td>3</td>
          <td>1</td>
          <td>3</td>
          <td>Heikkinen, Miss. Laina</td>
          <td>female</td>
          <td>26.0</td>
          <td>0</td>
          <td>0</td>
          <td>STON/O2. 3101282</td>
          <td>7.9250</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
        <tr>
          <th>3</th>
          <td>4</td>
          <td>1</td>
          <td>1</td>
          <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
          <td>female</td>
          <td>35.0</td>
          <td>1</td>
          <td>0</td>
          <td>113803</td>
          <td>53.1000</td>
          <td>C123</td>
          <td>S</td>
        </tr>
        <tr>
          <th>4</th>
          <td>5</td>
          <td>0</td>
          <td>3</td>
          <td>Allen, Mr. William Henry</td>
          <td>male</td>
          <td>35.0</td>
          <td>0</td>
          <td>0</td>
          <td>373450</td>
          <td>8.0500</td>
          <td>NaN</td>
          <td>S</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    NAs= pd.concat([dataset.isnull().sum()], axis=1, keys=["dataset"])
    NAs[NAs.sum(axis=1)>0]




.. raw:: html

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
          <th>dataset</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Age</th>
          <td>177</td>
        </tr>
        <tr>
          <th>Cabin</th>
          <td>687</td>
        </tr>
        <tr>
          <th>Embarked</th>
          <td>2</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    dataset["Age"]= dataset["Age"].fillna(dataset["Age"].mean())

.. code:: ipython3

    dataset["Embarked"]= dataset["Embarked"].fillna(dataset["Embarked"].mode()[0])

.. code:: ipython3

    dataset["Cabin"]= dataset["Cabin"].fillna(dataset["Embarked"].mode()[0])

.. code:: ipython3

    dataset["Pclass"]= dataset["Pclass"].apply(str)

.. code:: ipython3

    for col in dataset.dtypes[dataset.dtypes=="object"].index:
        for_dummy= dataset.pop(col)
        dataset= pd.concat([dataset,pd.get_dummies(for_dummy,prefix=col)],axis=1)
        dataset.head()

.. code:: ipython3

    labels = dataset.pop("Survived")
    from sklearn
    from sklearn.ensemble import RandomForestClassifier


::


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    ~\anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       2894             try:
    -> 2895                 return self._engine.get_loc(casted_key)
       2896             except KeyError as err:
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\index.pyx in pandas._libs.index.IndexEngine.get_loc()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    pandas\_libs\hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()
    

    KeyError: 'Survived'

    
    The above exception was the direct cause of the following exception:
    

    KeyError                                  Traceback (most recent call last)

    <ipython-input-33-d405d9262592> in <module>
    ----> 1 labels = dataset.pop("Survived")
          2 from sklearn.ensemble import RandomForestClassifier
    

    ~\anaconda3\lib\site-packages\pandas\core\frame.py in pop(self, item)
       4365         3  monkey        NaN
       4366         """
    -> 4367         return super().pop(item=item)
       4368 
       4369     @doc(NDFrame.replace, **_shared_doc_kwargs)
    

    ~\anaconda3\lib\site-packages\pandas\core\generic.py in pop(self, item)
        659 
        660     def pop(self, item: Label) -> Union["Series", Any]:
    --> 661         result = self[item]
        662         del self[item]
        663         if self.ndim == 2:
    

    ~\anaconda3\lib\site-packages\pandas\core\frame.py in __getitem__(self, key)
       2900             if self.columns.nlevels > 1:
       2901                 return self._getitem_multilevel(key)
    -> 2902             indexer = self.columns.get_loc(key)
       2903             if is_integer(indexer):
       2904                 indexer = [indexer]
    

    ~\anaconda3\lib\site-packages\pandas\core\indexes\base.py in get_loc(self, key, method, tolerance)
       2895                 return self._engine.get_loc(casted_key)
       2896             except KeyError as err:
    -> 2897                 raise KeyError(key) from err
       2898 
       2899         if tolerance is not None:
    

    KeyError: 'Survived'


