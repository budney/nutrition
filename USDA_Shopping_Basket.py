#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys

get_ipython().system('{sys.executable} -m pip install mysql.connector')
get_ipython().system('{sys.executable} -m pip install chart_studio')


# In[3]:


# Customize stuff globally
DAYS = 7

# Location of the USDA database (Len's Docker image)
PORT = 3306
HOST = 'usda'
AGE  = 51   # 1, 4, 9, 14, 19, 31, 51, or 70


# In[4]:


# Import libraries we're going to need.
import mysql.connector
import pandas as pd
import chart_studio.tools as tls
import numpy as np

# We're going to perform a Simplex optimization
from scipy.optimize import linprog


# In[5]:


# Function definitions for the rest of the workbook

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
    return(['food_code', 'food_desc', 'age_from', 'sex', 'type', 'enerc_kcal', 'fat', 'f18d2', 'procnt', 'chocdf', 'fibtg', 'pct_fibtg', 'pct_ca', 'pct_fe', 'pct_mg', 'pct_p',
    'pct_k', 'pct_na', 'pct_zn', 'pct_cu', 'pct_mn', 'pct_se', 'pct_vitc', 'pct_thia', 'pct_ribf',
    'pct_nia', 'pct_vitb6a', 'pct_fol', 'pct_choln', 'pct_vitb12', 'pct_vita_rae', 'pct_tocpha',
    'pct_vitd', 'pct_vitk1', 'pct_f18d2', 'pct_f18d3'])

def query(age = AGE):
    return("""
        SELECT * FROM contrib.food_dri_pct_view
        WHERE sex = 'male'
        AND   type = 'recommended'
        AND   age_from = %s
    """ % age)

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
        np.dot(result, data.loc[:, 'enerc_kcal'].to_numpy()),
        np.multiply(np.dot(result, data.loc[:, 'chocdf'].to_numpy()), 4),
        np.multiply(np.dot(result, data.loc[:, 'procnt'].to_numpy()), 4),
        np.multiply(np.dot(result, data.loc[:, 'fat'].to_numpy()), 9),
        np.multiply(np.dot(result, data.loc[:, 'f18d2'].to_numpy()), 9),
        np.dot(result, data.loc[:, 'fibtg'].to_numpy()),
    ]
    macros = pd.DataFrame(macros)
    macros.index = ['Calories', 'Carb Cal', 'Protein Cal', 'Fat Cal', 'Omega-6 Cal', 'Fiber gm']
    macros.columns = ['Value']
    return(macros.div(DAYS))

def get_micros(data, result):
    # Get the nutrients for the foods
    nutrients = data.loc[:, 'pct_fibtg':'pct_f18d3'].copy()
    nutrients = (nutrients.T * result).T
    nutrients = round(nutrients, 3)
    
    # Total them
    return(np.multiply(nutrients.sum(axis=0), 100/DAYS))

def get_rations(data, result):
    # Get the food list
    rations = data.loc[:, 'food_code':'food_desc']
    rations.reset_index(drop=True, inplace=True)
    
    # Get the amounts from the solution, converted to grams
    amounts = pd.DataFrame(np.multiply(result, 100))
#    amounts.reset_index(drop=True, inplace=True)
    
    # Add the nutrients for the foods
    nutrients = data.loc[:, 'enerc_kcal':'pct_f18d3'].copy()
    nutrients = (nutrients.T * result).T
    nutrients = round(nutrients, 3)
#    nutrients.reset_index(drop=True, inplace=True)

    # Combine them and label them
    rations = pd.concat((rations, amounts, nutrients), axis=1)
    cols = ['Code', 'Food', 'Amount (gm)']
    cols.extend(columns()[5:])
    rations.columns = cols
    
    # Sort by descending amounts
    rations = rations.sort_values('Amount (gm)', ascending=False)
    rations.reset_index(drop=True, inplace=True)
    return(rations)

def summarize_solution(data, solution):
    threshold = 0.01
    if solution.success:
        data = data[solution.x > threshold].reset_index(drop=True)
        X = solution.x[solution.x > threshold]
        
        print(get_macros(data, X))
        print()
        print(get_micros(data, X))
        print()
        print(get_rations(data, X))
    return


# In[6]:


# Function for adding a constraint on the total weight of specified foods
def add_min_weight(data, constraints=[], bounds=[], regex=r'', limits=[], min=None, max=6800):
    matched = data['food_code'].astype(str).str.match(regex)
    limits = [(0,max/100) if matched[i] else limits[i] for i in range(len(matched))]
    
    coefs = np.transpose([1 if matched[i] else 0 for i in range(len(matched))])
    
    # Minimum is 90% of the number given
    constraints = np.c_[constraints, np.multiply(coefs, -1)]
    bounds.append(-1 * 0.9 * min / 100)

    # Maximum is 110% of the number given
    constraints = np.c_[constraints, coefs]
    bounds.append( 1.1 * min / 100)
    
    return(constraints, bounds, limits)

# Function for adding constraints to the nutrition LP
def add_range(data, constraints=[], bounds=[], nutrient='enerc_kcal', min=None, max=None):
    if nutrient not in data.columns:
        return(constraints, bounds)
    
    coefs = np.transpose([data.loc[:, nutrient].to_numpy()])
    
    if min is not None:
        constraints = np.c_[constraints, np.multiply(coefs, -1)]
        bounds.append(-1 * min)
    
    if max is not None:
        constraints = np.c_[constraints, coefs]
        bounds.append(max)
    
    return(constraints, bounds)

# Function for adding percentage of calories
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


# # Fetch Food Data

# In[7]:


# Lookup food data from the DB
data = get_data(query())
data.columns = columns()

# Strip out any records with no nutritional value at all
data = data[np.linalg.norm(data.loc[:, 'pct_fibtg':'pct_f18d3'], axis=1) != 0]

# Strip out the foods that are missing cost data
#cost = data.loc[:, 'cost']
#data = data[np.isnan(cost) == False]

# Reset indices on the remaining data
data.reset_index(drop=True, inplace=True)

# Impute a glycemic index of 100 to foods that don't have one
#data.fillna(value={'glycemic_index': 100}, inplace=True)

# Impute 0 to all nutritional information
data.fillna(0, inplace=True)

# Look up the upper limits for nutrients, if known
upper_limits = get_data(upper_limit_query())
upper_limits.columns = ['tagname', 'amount']
for i in range(len(upper_limits)):
    upper_limits.loc[i, 'tagname'] = 'pct_' + upper_limits.loc[i, 'tagname']

print(data.shape)


# # Objective Function

# In[8]:


# Pick an objective function here:
# objective = data.loc[:, 'enerc_kcal'].to_numpy() # Minimize calories
# objective = [1 for row in constraints] # Minimize weight
# objective = np.multiply(data.loc[:, 'fibtg'].to_numpy(), -1) # Maximize fiber
objective = data.loc[:, 'chocdf'].to_numpy() # Minimize carbs
# objective = data.loc[:, 'fat'].to_numpy() # Minimize fat
# objective = np.multiply(data.loc[:, 'f18d2'].to_numpy(), -1) # Maximize Omega-6
# objective = data.loc[:, 'cost'].to_numpy() # Minimize cost
# objective = np.multiply(0.01, np.multiply(data.loc[:, 'glycemic_index'], data.loc[:, 'chocdf'])).to_numpy() # Min glycemic load


# # Nutritional Constraints

# In[9]:


# Require 100% of every nutrient with an RDA. Times 7, because this meal plan is for a week.
constraints = np.multiply( data.loc[:, 'pct_fibtg':'pct_f18d3'].to_numpy(), -1 + 0.1 )
bounds = [ -1*DAYS for row in constraints.T]

# Set calories between 1800 and 2100
constraints, bounds = add_range( data, constraints, bounds, 'enerc_kcal', min=1800*DAYS, max=2000*DAYS )

# Restrict nutrients that have upper limits
for i in range(len(upper_limits)):
    tag, amount = upper_limits.loc[i]
    constraints, bounds = add_range(data, constraints, bounds, tag, min=None, max=amount*DAYS)

# Restrict the remaining nutrients, because enough is as good as a feast
for tag in data.columns:
    if tag[:4] != 'pct_':
        continue
    if tag in upper_limits.loc[:, 'tagname'].to_numpy():
        continue
    constraints, bounds = add_range(data, constraints, bounds, tag, min=None, max=4*DAYS)


# # Macronutrient Ratios

# In[10]:


# Set protein between 10 and 35 percent of energy
#constraints, bounds = add_energy_percent_range(data, constraints, bounds, 'chocdf', mult=4, min=.40,     max=.65)
#constraints, bounds = add_energy_percent_range(data, constraints, bounds, 'procnt', mult=4, min=.099999, max=.35)
#constraints, bounds = add_energy_percent_range(data, constraints, bounds, 'fat',    mult=9, min=.20,     max=.35)
#constraints, bounds = add_energy_percent_range(data, constraints, bounds, 'f18d2',  mult=9, min=.01,     max=.10)


# # Food Group Constraints
# 
# _AKA "Market Basket" constraints, such as "X lbs of potato products."_

# In[11]:


# Set an initial constraint eliminating every food
limits = [(0, 0) for i in range(len(objective))]

# Orange vegetables: 0.88 lbs, 400 gm
constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'73[1234]', limits=limits, min=400)

# Dark-green vegetables: 1.12 lbs, 508 gm
constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'72[12]', limits=limits, min=508)

# Legumes: 2.64 lbs, 1,197 gm
constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'41[1234]', limits=limits, min=1197)

# Potatoes: 1.61 lbs, 730 gm
#constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'7110([124]|3[01])', limits=limits, min=730)

# Other veggies: 3.39 lbs, 1,538 gm
constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'7(4[12]|5(2|1(0([0-8]|9[06])|1[0-8]|[2-4])))', limits=limits, min=1538)

# Fruits: 7 lbs, 3,175 gm
# constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'6[123]1|632', limits=limits, min=3175, max=453)

# Fruit Juices: 1.68 lbs, 762 gm
# constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'6[14]2|641', limits=limits, min=762)

# Whole grains (except cereal): 2.39 lbs, 1,084 gm
# constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'513|522|551|5814[67]', limits=limits, min=1084)

# Whole grain cereals: 0.10 lbs, 45 gm
# constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'562', limits=limits, min=45)

# Whole grain snacks: 0.20 lbs, 90 gm
# constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'544', limits=limits, min=90)

# Non-whole grains: 2.04 lbs, 925 gm
# constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'511', limits=limits, min=925)

# Whole dairy: 0.39 lbs, 176 gm
constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'11111000|11115300|11116000|11211050|11411100|11411400', limits=limits, min=176)

# Lowfat dairy: 12.33 lbs, 5,593 gm
constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'11113000|11112110|11112210|11115000|11115100|11411200|11411300|11411410|11411420', limits=limits, min=5593)

# Cheese: 0.13 lbs, 60 gm
constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'14[0-4]', limits=limits, min=60)

# Dairy treats: 0.15 lbs, 68 gm
# constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'13', limits=limits, min=68)

# Poultry: 2.95 lbs, 1,338 gm
#constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'24[1-4]', limits=limits, min=1338)
# constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'241000[12]|241201[12]|241502[12]|24160110|242010[23]|242011[23]', limits=limits, min=1338)

# Fish: 0.42 lbs, 190 gm
#constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'26[13]', limits=limits, min=190)
constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'26101180|26109110|26117110|26121190|26137110|26149110|26153100|26153110|26158000|26305160|26317110|26319110', limits=limits, min=190)

# Meat: 1.04 lbs, 472 gm
# constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'21[0-6]|22[0-6]', limits=limits, min=472)

# Nuts & Seeds: 0.33 lbs, 150 gm
constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'4[23]', limits=limits, min=150)

# Eggs: 0.17 lbs, 77 gm
#constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'31', limits=limits, min=77)
constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'31', limits=limits, min=600)

# Lunchmeat: 0.11 lbs, 50 gm
# constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'252', limits=limits, min=50)

# Table fats: 0.47 lbs, 213 gm
#constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'8[23]1|83[12]', limits=limits, min=213)

# Sauces: 0.46 lbs, 209 gm
#constraints, bounds, limits = add_min_weight(data, constraints, bounds, regex=r'134|285|744', limits=limits, min=209)


# # Misc Constraints

# In[12]:


# Disallow specific foods I don't want to see in the results
for food_code in [63115010, 63115130, 91301030, 91301510, 91301080, 91304020, 64401000, 91304060, 55100010, 71205020, 75511010, 26205190, 75105500, 75124000, 81302050, 31110010, 31101010, 31108010, 26213100, 26123100, 26131100, 26118000, 26133100, 75236500, 75502500, 26311180, 26315180, 26315100, 24198500, 51136000, 58146315, 51182020, 58146215, 51182010, 63147010, 63147120, 23323100, 23326100, 75109550, 24302010, 23150270, 23340100, 23311200, 63208000, 63224000, 63205010, 23333100, 23324100, 23150200, 21401400, 11114320, 61113010, 11121300, 75102600, 26105121, 42403010, 26105160, 26105120, 26105190, 26105131, 26113160, 26113190, 26105130, 26137170, 26151160, 71901010, 26105140, 26105110, 25130000, 25150000, 25170110, 11114350, 25140110, 25170210, 26151190, 26151123, 26151124, 26151122, 26113110, 26151110, 26151120, 26151121, 26149160, 26149121, 26100190, 26137190, 26149120, 26149110, 71980100, 24400020, 24401020, 26137180, 26151133, 75109400, 26151134, 26121100, 26151143, 26151144, 26151132, 75100500, 31111020, 31111000, 31111010]:
    index = np.where(data.loc[:, 'food_code'] == food_code)
    if index[0].size == 0:
        continue
    limits[index[0][0]] = (0,0)

# Limit cost to $10 per day
#constraints, bounds = add_range(data, constraints, bounds, 'cost', max=10*7)

# Limit total weight to 2.5 kilos
#constraints = np.c_[constraints, np.transpose(np.ones(len(constraints)))]
#bounds.append(25)

# Add extra fiber to our diet
constraints, bounds = add_range(data, constraints, bounds, 'fibtg', min=20, max=None)


# # Solve the Equation

# In[13]:


# Try solving that
result = linprog(objective, A_ub=constraints.T, bounds=limits, b_ub=bounds, options={"disp": True})
print()

# Print out the results
summarize_solution(data, result)


# In[14]:


rations = get_rations(data[result.x>0.01].reset_index(drop=True), result.x[result.x>0.01])
rations.to_csv('/mnt/rations.csv')
print(rations)


# In[15]:


#summarize_solution(data, result)
options = data[[x[1] > 0 for x in limits]].reset_index(drop=True)
options.to_csv('/mnt/food_options.csv')
options

