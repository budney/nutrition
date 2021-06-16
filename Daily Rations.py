#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sys

get_ipython().system('{sys.executable} -m pip install mysql.connector')
get_ipython().system('{sys.executable} -m pip install chart_studio')


# In[8]:


# Customize stuff globally

# Location of the USDA database (Len's Docker image)
PORT = 3306
HOST = 'usda'
# IP = ! netstat -r -n|egrep '^0.0.0.0'|awk '{print $2}'
# IP = IP[0]


# In[9]:


# Function definitions for the rest of the workbook
import mysql.connector
import pandas as pd
import chart_studio.plotly as py
from plotly.graph_objects import *
import plotly.tools as tls
import numpy as np

# Disable Pandas' annoying "future" warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def connect():
    return(mysql.connector.connect(
       host=HOST,
       port=PORT,
       user="root",
       passwd="admin",
       database="fndds",
    ))

def get_data(query):
    db = connect()
    cur = db.cursor()
    cur.execute(query)
    data = cur.fetchall()
    db.close()
    return(pd.DataFrame(data))
    
def columns():
    return(['food_code', 'food_desc', 'food_group_code', 'food_group_desc',
    'is_animal', 'is_dairy', 'is_red_meat', 'is_beef', 'is_other_red_meat',
    'is_white_meat', 'is_pork', 'is_poultry', 'is_sausage_or_organ_meat',
    'is_seafood', 'is_eggs', 'is_legume', 'is_nut_or_seed', 'is_bread',
    'is_other_grain_product', 'is_fruit', 'is_vegetable', 'is_white_potato',
    'is_other_vegetable', 'is_fat', 'is_sweetener',
    'pct_water', 'enerc_kcal', 'fat', 'f18d2', 'procnt', 'chocdf', 'fibtg', 'pct_fibtg', 'pct_ca', 'pct_fe', 'pct_mg', 'pct_p',
    'pct_k', 'pct_na', 'pct_zn', 'pct_cu', 'pct_mn', 'pct_se', 'pct_vitc', 'pct_thia', 'pct_ribf',
    'pct_nia', 'pct_vitb6a', 'pct_fol', 'pct_choln', 'pct_vitb12', 'pct_vita_rae', 'pct_tocpha',
    'pct_vitd', 'pct_vitk1', 'pct_f18d2', 'pct_f18d3','glycemic_index', 'cost'])

def query():
    return("SELECT * FROM contrib.food_dri_view;")

def upper_limit_query():
    return("""
        SELECT
            tagname, (tolerable_upper.amount / recommended.amount) AS pct_tolerable_upper
        FROM (
                SELECT age_from, sex, tagname, amount
                FROM dietary_reference_intake.dietary_reference_intake
                WHERE type = 'tolerable_upper'
                AND age_from = 31
                AND sex = 'Male'
        ) AS tolerable_upper
        JOIN dietary_reference_intake.dietary_reference_intake AS recommended USING(age_from, sex, tagname)
        WHERE
            recommended.type = 'recommended'
            AND (tolerable_upper.amount / recommended.amount) > 1
    """)

def get_macros(data, result):
    macros = [
        np.dot(result.x, data.loc[:, 'enerc_kcal'].to_numpy()),
        np.multiply(np.dot(result.x, data.loc[:, 'chocdf'].to_numpy()), 4),
        np.multiply(np.dot(result.x, data.loc[:, 'procnt'].to_numpy()), 4),
        np.multiply(np.dot(result.x, data.loc[:, 'fat'].to_numpy()), 9),
        np.multiply(np.dot(result.x, data.loc[:, 'f18d2'].to_numpy()), 9),
        np.dot(result.x, data.loc[:, 'fibtg'].to_numpy()),
    ]
    macros = pd.DataFrame(macros)
    macros.index = ['Calories', 'Carb Cal', 'Protein Cal', 'Fat Cal', 'Omega-6 Cal', 'Fiber gm']
    macros.columns = ['Value']
    return(macros)

def get_micros(data, result):
    # Get the nutrients for the foods
    nutrients = data.loc[:, 'pct_fibtg':'pct_f18d3'].copy()
    nutrients = (nutrients.T * result.x).T
    nutrients = nutrients[result.x>0]
    nutrients = round(nutrients, 3)
    nutrients.reset_index(drop=True, inplace=True)
    
    # Total them
    return(np.multiply(nutrients.sum(axis=0), 100))

def get_rations(data, result):
    # Get the food list
    rations = data.loc[:, 'food_code':'food_desc'][result.x>0]
    rations.reset_index(drop=True, inplace=True)
    
    # Get the amounts from the solution, converted to grams
    amounts = pd.DataFrame(np.multiply(result.x[result.x>0], 100))
    amounts.reset_index(drop=True, inplace=True)
    
    # Add the nutrients for the foods
    nutrients = data.loc[:, 'enerc_kcal':'pct_f18d3'].copy()
    nutrients = (nutrients.T * result.x).T
    nutrients = nutrients[result.x>0]
    nutrients = round(nutrients, 3)
    nutrients.reset_index(drop=True, inplace=True)

    # Combine them and label them
    rations = pd.concat((rations, amounts, nutrients), axis=1)
    cols = ['Code', 'Food', 'Amount (gm)']
    cols.extend(columns()[26:57])
    rations.columns = cols
    
    # Sort by descending amounts
    rations = rations.sort_values('Amount (gm)', ascending=False)
    rations.reset_index(drop=True, inplace=True)
    return(rations)

def summarize_solution(data, solution):
    if solution.success:
        print(get_macros(data, solution))
        print()
        print(get_micros(data, solution))
        print()
        print(get_rations(data, solution))
    return


# In[10]:


# Functions for adding constraints to the nutrition LP
def add_range(data, constraints, nutrient='enerc_kcal', min=None, max=None):
    if nutrient not in data.columns:
        return(constraints)
    
    coefs = data.loc[:, nutrient].append(pd.Series(0, index=['bounds']), ignore_index=False).fillna(0)
    coefs.index = constraints.index
    
    # Validate the coefficients
    if coefs.isna().any() or np.isinf(coefs).any() or None in coefs:
        print('Bad values for nutrient ' + nutrient)
        print(coefs)
        return(constraints)
        
    if min is not None:
        coefs['bounds'] = -1*min
        constraints['min_' + nutrient] = coefs.copy()
    
    if max is not None:
        coefs['bounds'] = max
        constraints['max_' + nutrient] = coefs
    
    return(constraints)

# Functions for adding percentage of calories
def add_energy_percent_range(data, constraints=[], bounds=[], nutrient='chocdf',  mult=4, min=None, max=None):
    nut_coefs = np.multiply( np.transpose([data.loc[:, nutrient].to_numpy()]), mult )
    cal_coefs = np.transpose([data.loc[:, 'enerc_kcal'].to_numpy()])
    
    if min is not None:
        coefs = np.subtract(np.multiply(cal_coefs, min), nut_coefs)
        constraints = np.c_[constraints, coefs]
        bounds.append(0)
    
    if max is not None:
        coefs = np.subtract(nut_coefs, np.multiply(cal_coefs, max))
        constraints = np.c_[constraints, coefs]
        bounds.append(0)
    
    return(constraints, bounds)


# In[11]:


# Lookup food data from the DB
data = get_data(query())
data.columns = columns()

# Strip out any records with no nutritional value at all
data = data[np.linalg.norm(data.loc[:, 'pct_fibtg':'pct_f18d3'], axis=1) != 0]
data.reset_index(drop=True, inplace=True)

# Look up the upper limits for nutrients, if known
upper_limits = get_data(upper_limit_query())
upper_limits.columns = ['tagname', 'amount']
for i in range(len(upper_limits)):
    upper_limits.loc[i, 'tagname'] = 'pct_' + upper_limits.loc[i, 'tagname']

print(data.shape)


# In[12]:


# Perform a Simplex optimization
from scipy.optimize import linprog

# Pick an objective function here:
# objective = [1 for row in constraints] # Minimize weight
# objective = np.multiply(data.loc[:, 'fibtg'].to_numpy(), -1) # Maximize fiber
# objective = data.loc[:, 'enerc_kcal'].to_numpy() # Minimize calories
objective = data.loc[:, 'chocdf'].to_numpy() # Minimize carbs
# objective = data.loc[:, 'fat'].to_numpy() # Minimize fat
# objective = np.multiply(data.loc[:, 'f18d2'].to_numpy(), -1) # Maximize Omega-6
# objective = data.loc[:, 'cost'].to_numpy() # Minimize cost
# objective = np.multiply(0.01, np.multiply(data.loc[:, 'glycemic_index'], data.loc[:, 'chocdf'])).to_numpy() # Min glycemic load

# Require 100% of every nutrient with an RDA
constraints = data.loc[:, 'pct_ca':'pct_f18d3'].fillna(0)
constraints = constraints * -1 # (-1 + 0.1)

# Label the constraints while we're at it
#constraints.columns = data.loc[:, 'pct_fibtg':'pct_f18d3'].columns
constraints.columns = data.loc[:, 'pct_ca':'pct_f18d3'].columns
constraints.index = data['food_desc']

# Add bounds to our constraints, as a bottom row
constraints = constraints.append(pd.Series(-1, index=constraints.columns, name='bounds'), ignore_index=False)

# Set calories between 1800 and 2100
# constraints = add_range(data, constraints, 'enerc_kcal', min=1800, max=2000)

# Restrict nutrients that have upper limits
for i in range(len(upper_limits)):
    tag, amount = upper_limits.loc[i]
    continue
    constraints = add_range(data, constraints, tag, min=None, max=amount)

# Restrict the remaining nutrients, because enough is as good as a feast
for tag in data.columns:
    if tag[:4] != 'pct_':
        continue
    if tag in upper_limits.loc[:, 'tagname'].to_numpy():
        continue
    constraints = add_range(data, constraints, tag, min=None, max=4)

# Limit total weight to 2.5 kilos
#constraints = np.c_[constraints, np.transpose(np.ones(len(constraints)))]
#bounds.append(25)

# Add extra fiber to our diet
#constraints = add_range(data, constraints, 'fibtg', min=5, max=None)

# Set protein between 10 and 35 percent of energy
#constraints = add_energy_percent_range(data, constraints, 'chocdf', mult=4, min=.40,     max=.65)
#constraints = add_energy_percent_range(data, constraints, 'procnt', mult=4, min=.099999, max=.35)
#constraints = add_energy_percent_range(data, constraints, 'fat',    mult=9, min=.20,     max=.35)
#constraints = add_energy_percent_range(data, constraints, 'f18d2',  mult=9, min=.01,     max=.10)

# Disallow more than a pound of any one food
limits = [(0, 4.5) for i in range(len(objective))]

# Exclude weird foods as they come up
for food_code in [63115010, 63115130, 91301030, 91301510, 91301080, 91304020, 64401000, 91304060, 55100010, 71205020, 75511010, 26205190, 75105500, 75124000, 81302050, 31110010, 31101010, 31108010, 26213100, 26123100, 26131100, 26118000, 26133100, 75236500, 75502500, 26311180, 26315180, 26315100]:
    index = np.where(data.loc[:, 'food_code'] == food_code)
    if not index[0]:
        continue
    limits[index[0][0]] = (0,0)

# Try solving that
result = linprog(objective, A_ub=constraints[:-1].to_numpy().T, bounds=limits, b_ub=constraints.loc['bounds', :].to_numpy(), options={"disp": True})
print()

# Print out the results
summarize_solution(data, result)


# In[13]:


rations = get_rations(data, result)
# rations.to_csv('/mnt/rations.csv')
print(rations)


# In[ ]:




