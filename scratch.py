# -> pandas can handle different file formats
#pandas basics
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

web_stats = {'Day' : [1,2,3,4,5,6],
             'Visitors' : [43,53,34,45,64,34],
             'Bounce_Rate' : [65,72,62,64,54,66]}

df = pd.DataFrame(web_stats)

'''
#make a data frame
print(df)
print(df.head())
print(df.tail)
print(df.tail(2))
'''

#choose the index

'''
print(df.set_index('Day'))      #returned a new data frame
print(df.head())
'''

#df = df.set_index('Day')
#df2 = df.set_index('Day')
#print(df2.head())

'''
df.set_index('Day', inplace=True)
print(df.head())
'''

#print(df['Visitors'])       #can handle a space
#print(df.Visitors)

#reference multiple columns

#print(df[['Bounce_Rate','Visitors']])

print(df.Visitors.tolist())

print(np.array(df[['Bounce_Rate','Visitors']]))

df2 = pd.DataFrame(np.array(df[['Bounce_Rate','Visitors']]) )
print(df2)

///////////////////////////////////////////////////////////////
#basics
import pandas as pd

'''
df = pd.read_csv('ZILLOW-N151_LPCSSF.csv')
#print(df.head())
df.set_index('Date', inplace=True)
df.to_csv('newcsv2.csv')
#print(df.head())
df = pd.read_csv('newcsv2.csv')     #csv has no attribute index
print(df.head())
df = pd.read_csv('newcsv2.csv', index_col=0)     #csv has no attribute index #set index during read
print(df.head())
df.columns = ['Austin_HPI']
print(df.head())
df.to_csv('newcsv3.csv')

df.to_csv('newcsv4.csv', header=False)  #with no header
#read csv with no header

df = pd.read_csv('newcsv4.csv', names=['Date', 'Austin_HPI'], index_col=0)
print(df.head())



#convert into a html
df.to_html('example.html')

'''
df = pd.read_csv('newcsv4.csv', names=['Date','Austin_HPI'])
print(df.head())
df.rename(columns={'Austin_HPI':'77006_HPI'}, inplace=True)
print(df.head())

///////////////////////////////////////////////////////////////////////
#p4
#little bit about real state analysis
# Building Dataset
import quandl
import pandas as pd

api_key = "iyqJBVtH-Njpn3ATpyzD"

df = quandl.get('FMAC/HPI_AK', authtoken=api_key )
# print(df.head())
# df.to_csv('p4.csv')


fiddy_states = pd.read_html("https://simple.wikipedia.org/wiki/List_of_U.S._states")
# fiddy_states is a list of data frames
# this is a list:
print(fiddy_states)


#this is a dataframe:
print(fiddy_states[0])

#this is a column
print(fiddy_states[0][0])

for abbv in fiddy_states[0][0][1:]:
    print("FMAC/HPI_" + str(abbv))


//////////////////////////////////////////////////
# p5
# concatinating and appending
# Combining DataFrames
import pandas as pd

import pandas as pd

df1 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                   index = [2001, 2002, 2003, 2004])

df2 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                   index = [2005, 2006, 2007, 2008])

df3 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'Low_tier_HPI':[50, 52, 50, 53]},
                   index = [2001, 2002, 2003, 2004])

'''
concat = pd.concat([df1,df2])
print(concat)

concat = pd.concat([df1,df2,df3])
print(concat)

df4 = df1.append(df3)
print(df4)
# Result is just Creating three new columns but we want to append into the same columns so next we will use the columns name
s = pd.Series([80,2,50])

df4 = df1.append(s ,ignore_index=True)
print(df4)
'''

s = pd.Series([80,2,50], index = ['HPI','Int_rate','US_GDP_Thousands'] )

df4 = df1.append(s, ignore_index=True)
print(df4)
# doubt give seriesa a name
/////////////////////////////////////////////////////////
# p6
# Joining and Merging
# difference one cares about index other doesn't

import pandas as pd

df1 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                   index = [2001, 2002, 2003, 2004])

df2 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                   index = [2005, 2006, 2007, 2008])

df3 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Unemployment':[7, 8, 9, 6],
                    'Low_tier_HPI':[50, 52, 50, 53]},
                   index = [2001, 2002, 2003, 2004])

#print(pd.merge(df1,df2, on=['HPI']))

'''
# output
  HPI  Int_rate_x  US_GDP_Thousands_x  Int_rate_y  US_GDP_Thousands_y
0   80           2                  50           2                  50
1   85           3                  55           3                  55
2   85           3                  55           2                  55
3   85           2                  55           3                  55
4   85           2                  55           2                  55
5   88           2                  65           2                  65
'''

# print(pd.merge(df1,df2, on=['HPI','Int_rate']))
'''
# output
   HPI  Int_rate  US_GDP_Thousands_x  US_GDP_Thousands_y
0   80         2                  50                  50
1   85         3                  55                  55
2   88         2                  65                  65
3   85         2                  55                  55
'''

'''
#input
df1.set_index('HPI', inplace=True)
df3.set_index('HPI', inplace=True)

#joined = df1.join(df3)
print(joined)
'''
'''
#output
     Int_rate  US_GDP_Thousands  Low_tier_HPI  Unemployment
HPI                                                        
80          2                50            50             7
85          3                55            52             8
85          3                55            53             6
85          2                55            52             8
85          2                55            53             6
88          2                65            50             9
'''

# Joining and Merging
# merge ignores index
# join respects index

import pandas as pd


df1 = pd.DataFrame({
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55],
                    'Year':[2001, 2002, 2003, 2004]
                    })

df3 = pd.DataFrame({
                    'Unemployment':[7, 8, 9, 6],
                    'Low_tier_HPI':[50, 52, 50, 53],
                    'Year':[2001, 2003, 2004, 2005]})


'''
#input
merged = pd.merge(df1,df3 ,on=['Year'])
print(merged)
'''

'''
#output
 Int_rate  US_GDP_Thousands  Year  Low_tier_HPI  Unemployment
0         2                50  2001            50             7
1         2                65  2003            52             8
2         2                55  2004            50             9
'''

'''
merged = pd.merge(df1,df3 ,on=['Year'])
merged.set_index('Year', inplace = True)
print(merged)
'''
'''
#output
      Int_rate  US_GDP_Thousands  Low_tier_HPI  Unemployment
Year                                                        
2001         2                50            50             7
2003         2                65            52             8
2004         2                55            50             9
'''

'''
merged = pd.merge(df1,df3 ,on=['Year'],how ='left')
merged.set_index('Year', inplace = True)
print(merged)
'''

'''
#output
      Int_rate  US_GDP_Thousands  Low_tier_HPI  Unemployment
Year                                                        
2001         2                50          50.0           7.0
2002         3                55           NaN           NaN
2003         2                65          52.0           8.0
2004         2                55          50.0           9.0
'''

'''
merged = pd.merge(df1,df3 ,on=['Year'],how ='left') # df1
merged.set_index('Year', inplace = True)
print(merged)

merged = pd.merge(df1,df3 ,on=['Year'],how ='right') #df3
merged.set_index('Year', inplace = True)
print(merged)

merged = pd.merge(df1,df3 ,on=['Year'],how ='outer')    #union
merged.set_index('Year', inplace = True)
print(merged)

merged = pd.merge(df1,df3 ,on=['Year'],how ='inner')    #intersection
merged.set_index('Year', inplace = True)
print(merged)

'''
'''
#output
      Int_rate  US_GDP_Thousands  Low_tier_HPI  Unemployment
Year                                                        
2001         2                50          50.0           7.0
2002         3                55           NaN           NaN
2003         2                65          52.0           8.0
2004         2                55          50.0           9.0
      Int_rate  US_GDP_Thousands  Low_tier_HPI  Unemployment
Year                                                        
2001       2.0              50.0            50             7
2003       2.0              65.0            52             8
2004       2.0              55.0            50             9
2005       NaN               NaN            53             6
      Int_rate  US_GDP_Thousands  Low_tier_HPI  Unemployment
Year                                                        
2001       2.0              50.0          50.0           7.0
2002       3.0              55.0           NaN           NaN
2003       2.0              65.0          52.0           8.0
2004       2.0              55.0          50.0           9.0
2005       NaN               NaN          53.0           6.0
      Int_rate  US_GDP_Thousands  Low_tier_HPI  Unemployment
Year                                                        
2001         2                50            50             7
2003         2                65            52             8
2004         2                55            50             9
'''


merged = pd.merge(df1,df3 ,on=['Year'],how ='inner')    #intersection
merged.set_index('Year', inplace = True)
print(merged)

# concat and append to increase the dataframe
//////////////////////////////////////////////////////////////
# p7
# Pickling
import quandl
import pandas as pd
import pickle

# Not necessary, I just do this so I do not show my API key.
api_key = open('quandlapikey.txt', 'r').read()


def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]


def grab_initial_state_data():
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = quandl.get(query, authtoken=api_key)
        df.rename(columns={'Value': abbv}, inplace=True)  # quandl returns one column with name "Value"
        print(query)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    pickle_out = open('fiddy_states.pickle', 'wb')  #write bytes
    pickle.dump(main_df, pickle_out)
    pickle_out.close()


#grab_initial_state_data()
pickle_in = open('fiddy_states.pickle','rb')
HPI_data = pickle.load(pickle_in)
#print(HPI_data)

# panda's pickle is faster with very large data

HPI_data.to_pickle('pickle.pickle')
HPI_data2 = pd.read_pickle('pickle.pickle')
print(HPI_data2)

////////////////////////////////////////////////
# p8
# pandas has tons of calculations



#Create Correlation data
import pandas as pd

import quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')


# Not necessary, I just do this so I do not show my API key.
api_key = open('quandlapikey.txt', 'r').read()


def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]


def grab_initial_state_data():
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = quandl.get(query, authtoken=api_key)
        df.rename(columns={'Value': abbv}, inplace=True)  # quandl returns one column with name "Value"
        #df = df.pct_change()
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0
        print(query)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    pickle_out = open('fiddy_states3.pickle', 'wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

#grab_initial_state_data()
HPI_data = pd.read_pickle('fiddy_states3.pickle')

#modify columns
'''
HPI_data['TX'] = HPI_data['TX'] * 2
print(HPI_data['TX'].head())
'''

HPI_data.plot()
plt.legend().remove()       #shows scaling
plt.show()




#Create Correlation data
import pandas as pd

import quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')


# Not necessary, I just do this so I do not show my API key.
api_key = open('quandlapikey.txt', 'r').read()


def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]


def grab_initial_state_data():
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = quandl.get(query, authtoken=api_key)
        df.rename(columns={'Value': abbv}, inplace=True)  # quandl returns one column with name "Value"
        #df = df.pct_change()
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0
        print(query)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    pickle_out = open('fiddy_states3.pickle', 'wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

#grab_initial_state_data()
HPI_data = pd.read_pickle('fiddy_states3.pickle')

def HPI_Benchmark():
    df = quandl.get("FMAC/HPI_USA", authtoken=api_key)
    df.rename(columns={'Value': 'United States'}, inplace=True)  # quandl returns one column with name "Value"
    df["United States"] = (df["United States"]-df["United States"][0]) / df["United States"][0] * 100.0
    print(df.head())
    return df

fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

HPI_data = pd.read_pickle('fiddy_states3.pickle')
benchmark = HPI_Benchmark()
HPI_data.plot(ax=ax1)
benchmark.plot(color='k',ax=ax1, linewidth=10)

plt.legend().remove()
plt.show()



#Create Correlation data
import pandas as pd

import quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')


# Not necessary, I just do this so I do not show my API key.
api_key = open('quandlapikey.txt', 'r').read()


def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]


def grab_initial_state_data():
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = quandl.get(query, authtoken=api_key)
        df.rename(columns={'Value': abbv}, inplace=True)  # quandl returns one column with name "Value"
        #df = df.pct_change()
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0
        print(query)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    pickle_out = open('fiddy_states3.pickle', 'wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

#grab_initial_state_data()
HPI_data = pd.read_pickle('fiddy_states3.pickle')

def HPI_Benchmark():
    df = quandl.get("FMAC/HPI_USA", authtoken=api_key)
    df.rename(columns={'Value': 'United States'}, inplace=True)  # quandl returns one column with name "Value"
    df["United States"] = (df["United States"]-df["United States"][0]) / df["United States"][0] * 100.0
    print(df.head())
    return df

'''
fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))
'''

HPI_data = pd.read_pickle('fiddy_states3.pickle')
'''
benchmark = HPI_Benchmark()
HPI_data.plot(ax=ax1)
benchmark.plot(color='k',ax=ax1, linewidth=10)

plt.legend().remove()
plt.show()
'''

HPI_State_Correlation = HPI_data.corr()
print(HPI_State_Correlation)

print(HPI_State_Correlation.describe())




/////////////////////////////////////
#p9

# Resampling
# df.rename(columns={'Value': 'United States'}, inplace=True)  # quandl returns one column with name "Value"
import pandas as pd

import quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')


# Not necessary, I just do this so I do not show my API key.
api_key = open('quandlapikey.txt', 'r').read()


def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]


def grab_initial_state_data():
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = quandl.get(query, authtoken=api_key)
        df.rename(columns={'Value': abbv}, inplace=True)  # quandl returns one column with name "Value"
        #df = df.pct_change()
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0
        print(query)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    pickle_out = open('fiddy_states3.pickle', 'wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

def HPI_Benchmark():
    df = quandl.get("FMAC/HPI_USA", authtoken=api_key)
    df.rename(columns={'Value': 'United States'}, inplace=True)  # quandl returns one column with name "Value"
    df["United States"] = (df["United States"]-df["United States"][0]) / df["United States"][0] * 100.0
    print(df.head())
    return df


#grab_initial_state_data()
fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

HPI_data = pd.read_pickle('fiddy_states3.pickle')



HPI_data['TX'].plot(ax = ax1)
TX1yr = HPI_data['TX'].resample('A')
TX1yr.plot(color='k',ax=ax1)
plt.legend().remove()
plt.show()


# Resampling
# df.rename(columns={'Value': 'United States'}, inplace=True)  # quandl returns one column with name "Value"
import pandas as pd

import quandl
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')


# Not necessary, I just do this so I do not show my API key.
api_key = open('quandlapikey.txt', 'r').read()


def state_list():
    fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return fiddy_states[0][0][1:]


def grab_initial_state_data():
    states = state_list()

    main_df = pd.DataFrame()

    for abbv in states:
        query = "FMAC/HPI_" + str(abbv)
        df = quandl.get(query, authtoken=api_key)
        df.rename(columns={'Value': abbv}, inplace=True)  # quandl returns one column with name "Value"
        #df = df.pct_change()
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0
        print(query)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    pickle_out = open('fiddy_states3.pickle', 'wb')
    pickle.dump(main_df, pickle_out)
    pickle_out.close()

def HPI_Benchmark():
    df = quandl.get("FMAC/HPI_USA", authtoken=api_key)
    df.rename(columns={'Value': 'United States'}, inplace=True)  # quandl returns one column with name "Value"
    df["United States"] = (df["United States"]-df["United States"][0]) / df["United States"][0] * 100.0
    print(df.head())
    return df


#grab_initial_state_data()
fig = plt.figure()
ax1 = plt.subplot2grid((1,1), (0,0))

HPI_data = pd.read_pickle('fiddy_states3.pickle')

TX1yr = HPI_data['TX'].resample('A',how = 'mean')   #how means annually yearly#resample('A').mean()
print(TX1yr.head())

HPI_data['TX'].plot(ax = ax1, label='Monthly TX HPI')
TX1yr.plot(ax=ax1)

plt.legend(loc=4)
plt.show()



















