#!/usr/bin/env python
# coding: utf-8

# ***Introduction***
# 
# We have a dataset known as company_financial_data.csv, which contains essential information on various companies' financial metrics. This dataset includes various financial indicators such as Price, Price/Earnings ratios, Dividend Yields, Earnings per Share, and 52 Week High/Low prices, as well as Market Capitalization, EBITDA, Price/Sales, and Price/Book ratios across different sectors.
# 
# The task is to use SQL (SQL section in this Project) for preliminary data selection and summarization, followed by Python for more granular analysis and the creation of visual insights.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


# In[2]:


company_data = pd.read_csv("company_financial.csv")

display(company_data)


# In[3]:


company_data.isnull().sum().sort_values(ascending=False)


# In[4]:


company_data.info()


# In[5]:


company_data.describe()


# In[6]:


company_data['Market Cap'] = company_data['Market Cap'].astype(float)


# **Interactive Financial Correlation Analysis**
# 
# This task involves a sophisticated approach to visualizing financial data correlations, leveraging the power of Python's data analysis and visualization libraries. The resultant interactive heatmap is a valuable tool for financial analysts, investors, and data scientists, offering insights into how different financial metrics relate to each other in the business landscape.📈🖥️🔬
# 
# Objective: To perform an in-depth correlation analysis of financial metrics from a dataset of company data using Python, Pandas, and Plotly. The aim is to identify and visualize the relationships between various financial indicators, such as market capitalization, earnings per share, and dividend yield.

# In[7]:


numeric_columns = ['Price', 'PE_Ratio', 'Dividend Yield','High 52 Week', 'Low 52 Week', 'Market Cap', 'EBITDA', 'Price/Book', 'PB_PE_Ratio', 'Market_Cap_Ranking']

comp_corr = company_data[numeric_columns].corr()

print("Correlation Matrix: ")
print(comp_corr)


# In[8]:


fig = px.imshow(comp_corr, text_auto=True, aspect="auto", color_continuous_scale="bluered")

fig.update_traces(hovertemplate='x: %{x}<br>y: %{y}<br>corr coeff: %{z:.2f}')

fig.update_coloraxes(colorbar_title="Corr Coeff")

fig.update_layout(title="Heatmap",height=800, width=1000, font=dict(family="Arial, sans-serif", size=12, color='black'))

fig.show()


# **Financial Data Clustering and 3D Visualization**
# 
# This task encapsulates a blend of advanced data analysis, machine learning clustering techniques, and dynamic 3D visualization. It offers valuable insights into how companies group together based on financial metrics, serving as an essential tool for market analysis and investment strategy development.📈🖥️
# 
# Objective: To perform a clustering analysis of key financial metrics from various companies using Python's machine learning and visualization libraries. The goal is to segment the companies into distinct clusters based on their financial characteristics and visualize these clusters in a 3D space.

# In[9]:


#Standardizing the selected features to ensure they're on the same scale, which is crucial for effective clustering

from sklearn.preprocessing import StandardScaler

features_clustering = ['Price','PE_Ratio',"Market Cap", "Earnings/Share", "Dividend Yield"]

scaler = StandardScaler()

scaled_company_data = scaler.fit_transform(company_data[features_clustering])


# In[10]:


#Applying the K-Means clustering algorithm to segment the data into five clusters

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5,random_state=42,n_init=10)

clusters = kmeans.fit_predict(scaled_company_data)

company_data['Cluster'] = clusters


# In[11]:


fig = px.scatter_3d(company_data, x='Price',y='PE_Ratio',z='Market Cap',color='Cluster',
                    hover_name='Name',hover_data=['Earnings/Share','Dividend Yield','Sector'],
                   color_continuous_scale="turbo")

fig.show()


# **Sector-wise Financial Ratio Analysis and Visualization**
# 
# This task provides a comprehensive approach to analyzing and visualizing key financial ratios across sectors. It combines statistical analysis with interactive data visualization, making it an invaluable exercise for financial analysts, investors, and market researchers.📈📊💹
# 
# Objective: To analyze the Price to Earnings (PE) Ratios across different sectors using Python, SciPy, and Plotly, aiming to understand sectoral variations in this key financial metric. The task involves statistical testing to determine if there are significant differences in PE Ratios among sectors and visualizing these differences using an interactive bar chart.

# In[12]:


#Observing which is the sectors that we have to investigate & analyze furtherly

company_data['Sector'].unique()


# In[13]:


#Grouping our data by each sector to help us with our next statistical analysis

group_sector = company_data.groupby(by='Sector')
group_sector


# In[14]:


variances = company_data.groupby('Sector')['PE_Ratio'].var()

print(variances)


# **ANOVA Test 📊**
# 
# ANOVA (Analysis of Variance) test to check for significant differences in PE Ratios across different sectors. This step is crucial to determine if the average PE Ratio significantly varies from one sector to another.

# In[15]:


from scipy import stats

f_val, p_val = stats.f_oneway(company_data['PE_Ratio'][company_data['Sector']=='Consumer Discretionary'],
                             company_data['PE_Ratio'][company_data['Sector']=='Consumer Staples'],
                            company_data['PE_Ratio'][company_data['Sector']=='Energy'],
                              company_data['PE_Ratio'][company_data['Sector']=='Financials'],
                              company_data['PE_Ratio'][company_data['Sector']=='Health Care'],
                              company_data['PE_Ratio'][company_data['Sector']=='Industrials'],
                              company_data['PE_Ratio'][company_data['Sector']=='Information Technology'],
                              company_data['PE_Ratio'][company_data['Sector']=='Materials'],
                              company_data['PE_Ratio'][company_data['Sector']=='Real Estate'],
                              company_data['PE_Ratio'][company_data['Sector']=='Utilities']
                             )

print(f"ANOVA result on PE Ratios among sectors: F-statistic = {f_val:.2f}, p-value = {p_val:.4f}")


# **From the above results (p=0.007<0.05) we reject the null hypothesis of the ANOVA and conclude that there is a statistically significant difference between the means of the sectors.**
# 
# Otherwise, if the p-value was not less than .05 then we would fail to reject the null hypothesis and conclude that we do not have sufficient evidence to say that there is a statistically significant difference between the means of the sectors.

# In[16]:


#We create a new dataframe (pe_sector), calculating the mean, std and the size of each sector 
#from the above group by sector dataframe

pe_sector = group_sector['PE_Ratio'].agg([np.mean, np.std, np.size]).reset_index()
pe_sector


# **Confidence Interval Calculation 🔍**
# 
# Calculating the 95% confidence intervals for the mean PE Ratios of each sector, assuming a normal distribution. This gives an understanding of the range within which the true mean PE Ratio for each sector is likely to fall.

# In[17]:


confidence = 0.95

#calculating the confidence interval using stats.norm.ppf

pe_sector['conf_inter95_hi'] = pe_sector['mean'] + (pe_sector['std'] / np.sqrt(pe_sector['size']) * stats.norm.ppf((1 + confidence) / 2))
pe_sector['conf_inter95_lo'] = pe_sector['mean'] - (pe_sector['std'] / np.sqrt(pe_sector['size']) * stats.norm.ppf((1 + confidence) / 2))


# In[18]:


import plotly.graph_objects as go

colors = ['blue','yellow','green','red','orange','cyan','purple','magenta','springgreen','salmon']

fig = go.Figure(data=[
    go.Bar(
        x=pe_sector['Sector'],
        y=pe_sector['mean'],
        error_y=dict(
            type='data',
            array=pe_sector['conf_inter95_hi'] - pe_sector['mean'],
            arrayminus=pe_sector['mean'] - pe_sector['conf_inter95_lo']
        ),
        name='PE Ratio',
        marker_color=colors
    )
])

fig.update_layout(
    title='Confidence Intervals for Mean PE Ratio by Sector',
    xaxis_title='Sector',
    yaxis_title='Mean PE Ratio',
    template='plotly_white',
    autosize=True,
    width=800,
    height=600
)


fig.show()

