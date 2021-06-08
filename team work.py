"""
Die drei csv-Dateien enthalten Daten für Kryptowährungen für den Zeitraum von 01/01/2016 bis 
31/12/2020 von der Webseite https://www.cryptocompare.com/.
Cryptocompare aggregiert Preise und Stückzahlen von Kryptowährungen über mehrere Handelsplätze
und dient somit als adäquater Proxy für den Gesamtmarkt einer Kryptowährung.
Die Kryptowährungen sind die Top 100 Kryptowährungen mit der höchsten Marktkapitalisierung am
25/02/2021. 
Dies führt automatisch zu einem "survivorship bias", da wir z.B. nicht Kryptowährungen in unserem
Datensatz haben, die im Laufe des Beobachtungszeitraums z.B. liquidiert wurden.
Dieser Nachteil soll jedoch im Rahmen der Seminararbeit außer Acht gelassen werden.

- Top100_Prices.csv: enthält historische tägliche Closing Preise (in USD). Wenn der Preis NaN ist,
dann war die entsprechende Kryptowährung an diesem Handelstag nicht gelistet.
- Top100_Supply.csv: enthält historische tägliche Stückzahl einer Kryptowährung. 
- Top100_MCap.csv: ist die Marktkapitalisierung: Price x Supply

The following code snippets use pandas to compute some relevant variables that help your
for working on your seminar paper. See https://pandas.pydata.org/pandas-docs/stable/index.html.
"""

from numpy.core.fromnumeric import mean
import pandas as pd
from datetime import timedelta
from datetime import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats.mstats import winsorize
from pandas.tseries.offsets import Week
import statsmodels.api as sm
from numpy import NaN


pd.set_option("display.max_columns", 15)
##read data
prices = pd.read_csv("Top100_Prices.csv")
supply = pd.read_csv("Top100_Supply.csv")
mcap = pd.read_csv("Top100_MCap.csv")
tbill=pd.read_excel('1Month_Tbill.xlsx', usecols=["date","Bid Yield"],index_col=0)
tbilldiv52=tbill.div(52)
##tbilldiv52l=pd.melt(tbilldiv52.reset_index(), id_vars=["date"], value_vars=tbilldiv52.columns[0:], var_name="cur", value_name="mcapAgg_wide"))

"""
#every7thprice = prices[::7]
every7thprice = prices[prices.index % 6 == 1]
every7thprice.set_index("date", inplace=True)
every7thpriceret= every7thprice.apply(func = lambda x: x.shift(-1)/x -1)
momentumdf= every7thprice.apply(func= lambda x: x/x.shift(2) - 1) 
#momentumdflong=pd.melt(momentumdf, id_vars=momentumdf.index, value_vars=momentumdf[0:], var_name="cur", value_name="2weekreteveryweek")
##rankingdf= momentumdf.qcut()
#pricestest=prices.set_index()
"""
# Index as date. It is otherwise the same df as prices.
prices12=pd.read_csv("Top100_Prices1.csv",parse_dates=["date"], index_col="date")
mcap12 = pd.read_csv("Top100_MCap1.csv",parse_dates=["date"], index_col="date")






















mcapwkl=mcap12.resample("W-THU",kind="timestamp").mean()
mcl = mcap12.resample("W-THU",kind="timestamp").last()
##mcl2= mcap12.resample("2W-thu", kind="timestamp").last()
mcapAgg_wide=mcap12.resample("W", kind="timestamp").mean()
#star = pd.Timestamp("2016-01-29 00:00:00")
#end = pd.Timestamp("2021-01-04 00:00:00")

mcapAgg_long=pd.melt(mcapAgg_wide.reset_index(), id_vars=["date"], value_vars=mcapAgg_wide.columns[0:], var_name="cur", value_name="mcapAgg_wide")
mcapAgg_long["mcapAgg"]=mcapAgg_long.groupby("date")["mcapAgg_wide"].transform("sum")

#pivot table
lmcapwkl = pd.melt(mcapwkl.reset_index(), id_vars=["date"], value_vars=mcapwkl.columns[0:], var_name="cur", value_name="mcapwkl")

lmcaplastday= pd.melt(mcl.reset_index(), id_vars=["date"], value_vars=mcl.columns[0:], var_name="cur", value_name="mcl")

#prices12.replace(np.nan, 0)
#calculation of weekly ret with a daily pct change multiplied with each other for 1 Week. 
wkl_ret= prices12.pct_change(fill_method=None).resample("W-THU",kind="timestamp").agg(lambda x: (x+1).prod()-1)
wkl_ret2w= prices12.pct_change(fill_method=None).resample("2W-THU",closed="left", kind="timestamp").agg(lambda x: (x+1).prod()-1)
wkl_ret2w1= prices12.pct_change(fill_method=None).resample("2W-THU",closed="right", kind="timestamp").agg(lambda x: (x+1).prod()-1)

maxdprc= prices12.resample("W-THU",kind="timestamp").max()
##long format for maxdprc:
lmaxdprc = pd.melt(maxdprc.reset_index(),id_vars=["date"], value_vars=maxdprc.columns[0:], var_name="cur", value_name="maxdprc",ignore_index=True)
#print(lmaxdprc)
## Quantil sortieren:
lmaxdprc["qmaxdprc"]= lmaxdprc.groupby(['date'])["maxdprc"].transform(lambda x: pd.qcut(x, 5,labels=False, duplicates='drop'))

#print("qmaxdprc")
#print(qmaxdprc)
#print("********wkl_ret*********")
#print(wkl_ret)
##rng = pd.date_range(start, end, freq="W-thu")
##print(rng)

# using melt to pivot table to long format
lwkl_ret = pd.melt(wkl_ret.reset_index(),id_vars=["date"], value_vars=wkl_ret.columns[0:], var_name="cur", value_name="wklyret")
lwkl_ret2w = pd.melt(wkl_ret2w.reset_index(),id_vars=["date"], value_vars=wkl_ret2w.columns[0:], var_name="cur", value_name="wklyret2w")
lwkl_ret2wv2=pd.melt(wkl_ret2w1.reset_index(),id_vars=["date"], value_vars=wkl_ret2w1.columns[0:], var_name="cur", value_name="wklyret2wv2")
#print(lwkl_ret)
#replace zeros with nan value in preparation for qcut function
lwkl_ret['wklyret'].replace(to_replace=0, value=np.nan, inplace=True)
lwkl_ret2w['wklyret2w'].replace(to_replace=0, value=np.nan, inplace=True)
lwkl_ret2wv2['wklyret2wv2'].replace(to_replace=0, value=np.nan, inplace=True)
lwkl_ret2w.set_index("date", inplace=True)
lwkl_ret2wv2.set_index("date", inplace=True)

#bin column(2) into 5 quintiles 
##lwkl2=pd.qcut(lwkl_ret.iloc[:,2],5)
lwkl2= lwkl_ret.groupby(['date'])['wklyret'].transform(lambda x: pd.qcut(x, 5,labels=False, duplicates='drop'))


lwkl22w=lwkl_ret2w.groupby(['date'])['wklyret2w'].transform(lambda x: pd.qcut(x, 5,labels=False, duplicates='drop'))


wkl_data = pd.merge(lwkl_ret, lmcapwkl, how="left", on=["date", "cur"])
##quint2w war wohl alt
lwkl_ret2w["quint2w"]=lwkl22w

#write into fresh column
lwkl_ret["wklyquint"]=lwkl2
wkl_data["wklyquint"]=lwkl2

merg_2w=pd.merge(lwkl_ret, lwkl_ret2w, how="left", on=["date", "cur"])

merg_2wv2=pd.merge(merg_2w, lwkl_ret2wv2, how="left", on=["date", "cur"])
merg_2wv2["wklyret2w"]=merg_2wv2["wklyret2w"].fillna(0)
merg_2wv2["wklyret2wv2"]=merg_2wv2["wklyret2wv2"].fillna(0)
merg_2wv2["2wklyret2w+2wv2"]=merg_2wv2["wklyret2w"]+merg_2wv2["wklyret2wv2"]
merg_2wv2['2wklyret2w+2wv2'].replace(to_replace=0, value=np.nan, inplace=True)
wkq2cut=merg_2wv2.groupby(['date'])['2wklyret2w+2wv2'].transform(lambda x: pd.qcut(x, 5,labels=False, duplicates='drop'))

merg_2wv2["wkq2cut"]=wkq2cut

wkl_datav2=pd.merge(wkl_data, mcapAgg_long, how="left", on=["date", "cur"])

# TRY:
#retfw2_var=lwkl_ret2w.sort_values
retfw_var=lwkl_ret.sort_values("date", ascending=True).groupby("cur")["wklyret"].shift(periods=-1)
lwkl_ret["retfw"]=retfw_var
merg_2wv2["retfw"]=retfw_var
merg_2wv2=pd.merge(merg_2wv2, lmcaplastday, how="left", on=["date", "cur"])

wkl_datav2["retfw"]=retfw_var
#lmaxdprc["retfw"]=retfw_var
##lmaxdprc_merg_lmcaplastday= pd.merge(lmaxdprc, lmcaplastday, how="left", on=["date", "cur"])
#mcapwkls_var=wkl_datav2.sort_values("date", ascending=True).groupby("cur")["mcapwkl"].shift(periods=-1)
#wkl_datav2["mcapwkls"]=mcapwkls_var
mcapAggs_var=wkl_datav2.sort_values("date", ascending=True).groupby("cur")["mcapAgg"].shift(periods=-1)
wkl_datav2["mcapAggs"]=mcapAggs_var

mcapAggquint_var2=merg_2wv2.groupby(["date","wkq2cut"])["mcl"].sum()
merg_2wv3=pd.merge(merg_2wv2, mcapAggquint_var2, how="left", on=["date", "wkq2cut"])

wkl_datav2=pd.merge(wkl_datav2, lmcaplastday, how="left", on=["date", "cur"])

mcapAggquint_var=wkl_datav2.groupby("wklyquint")["mcl"].transform("sum")

wkl_datav2["mcapAggquint"]=mcapAggquint_var

merg_2wv3["fin2wret"]=merg_2wv3["retfw"]*(merg_2wv3["mcl_x"]/merg_2wv3["mcl_y"])

wkl_datav2["finwret"]= wkl_datav2["retfw"]*(wkl_datav2["mcl"] / wkl_datav2["mcapAggquint"])

portfolioreturnv=wkl_datav2.groupby("wklyquint")["finwret"].sum()
portfolioreturnsd=wkl_datav2.groupby("wklyquint")["finwret"].std()

print("portfolioreturn_1W_Momentum")
print(portfolioreturnv)

portfolioreturn_2W_momentumv=merg_2wv3.groupby(["date","wkq2cut"])["fin2wret"].sum()
portfolioreturn_2W_momentumfinal=portfolioreturn_2W_momentumv.groupby("wkq2cut").mean()
print("portfolioreturn_2W_momentumfinal")
print(portfolioreturn_2W_momentumfinal)

mergedmaxdprc = pd.merge(lmaxdprc,lwkl_ret, how="left", on=["date", "cur"])
mergedmaxdprc2= pd.merge(mergedmaxdprc, lmcaplastday, how="left", on=["date", "cur"])

#print("mergedmaxdprc")
#print(mergedmaxdprc)
##TO Do: gewichten:
mergedmaxdprc2.set_index("date", inplace=True)
mcapAggquint_var_maxdprc=mergedmaxdprc2.groupby(["date","qmaxdprc"])["mcl"].sum()

mergedmaxdprc2.to_excel("mergedmaxdprc2.xlsx")
mergedmaxdprc3= pd.merge(mergedmaxdprc2, mcapAggquint_var_maxdprc, how="left", on=["date", "qmaxdprc"])
##Gewichtete Rendite MaxDPRC Factor
mergedmaxdprc3["finmaxdret"]=mergedmaxdprc3["retfw"]*(mergedmaxdprc3["mcl_x"]/mergedmaxdprc3["mcl_y"])

#mergedmaxdprc2["mcapAggquint_maxdprc"]=mcapAggquint_var_maxdprc
meanfinmaxdret_maxdprc=mergedmaxdprc3.groupby(["date", "qmaxdprc"])["finmaxdret"].sum()
"""final maxdprc factor: meanfinmaxdretfinal
"""
meanfinmaxdretfinal=meanfinmaxdret_maxdprc.groupby("qmaxdprc").mean()
print("meanfinmaxdretfinal")
print(meanfinmaxdretfinal)
meanofquintile=lwkl_ret.groupby("wklyquint").mean()
##print("************ Number of distinct values in each group****")
##print(lwkl_ret.groupby("wklyquint").nunique())

#print("************ #meanofquintile# Mean of Weekly return grouped by wklyquint*****")
#print(meanofquintile)

"""
Last day price factor
"""
sizeprc= prices12.resample("W-THU",kind="timestamp").last()
sizeprcl=pd.melt(sizeprc.reset_index(), id_vars=["date"], value_vars=sizeprc.columns[0:], var_name="cur", value_name="sizeprc")
sizeprcl["logprice"]=np.log(sizeprcl["sizeprc"])
size_prc_qcut=sizeprcl.groupby(['date'])["sizeprc"].transform(lambda x: pd.qcut(x, 5,labels=False, duplicates='drop'))
sizeprcl["qulogp"]=size_prc_qcut
sizeprcl["retfw"]=retfw_var
mergedsizeprcl= pd.merge(sizeprcl, lmcaplastday, how="left", on=["date", "cur"])
mergedsizeprcl.set_index("date", inplace=True)
sizeprcmcapAggquint_var=mergedsizeprcl.groupby(["date","qulogp"])["mcl"].sum()
mergedsizeprcl1= pd.merge(mergedsizeprcl, sizeprcmcapAggquint_var, how="left", on=["date", "qulogp"])
mergedsizeprcl1["finmaxlogpret"]=mergedsizeprcl1["retfw"]*(mergedsizeprcl1["mcl_x"]/mergedsizeprcl1["mcl_y"])
sumweightlog=mergedsizeprcl1.groupby(["date", "qulogp"])["finmaxlogpret"].sum()
meanfinallogpret=sumweightlog.groupby("qulogp").mean()
"""
Last day price factor ENDE
"""
"""
Last day mcap factor ANFANG
"""
sizemcap= mcap12.resample("W-THU",kind="timestamp").last()
sizemcapl=pd.melt(sizemcap.reset_index(), id_vars=["date"], value_vars=sizemcap.columns[0:], var_name="cur", value_name="sizemcap")
sizemcapl["logmcap"]=np.log(sizemcapl["sizemcap"])
size_mcap_qcut=sizemcapl.groupby(['date'])["sizemcap"].transform(lambda x: pd.qcut(x, 5,labels=False, duplicates='drop'))
sizemcapl["qulogmcap"]=size_mcap_qcut
sizemcapl["retfw"]=retfw_var
mergedsizemcapl= pd.merge(sizemcapl, lmcaplastday, how="left", on=["date", "cur"])
mergedsizemcapl.set_index("date", inplace=True)
sizeMCAPmcapAggquint_var=mergedsizemcapl.groupby(["date","qulogmcap"])["mcl"].sum()
mergedsizemcapl1= pd.merge(mergedsizemcapl, sizeMCAPmcapAggquint_var, how="left", on=["date", "qulogmcap"])
mergedsizemcapl1["finmaxlogMCAPret"]=mergedsizemcapl1["retfw"]*(mergedsizemcapl1["mcl_x"]/mergedsizemcapl1["mcl_y"])
sumweightlogmcap=mergedsizemcapl1.groupby(["date", "qulogmcap"])["finmaxlogMCAPret"].sum()
meanfinallogMCAPret=sumweightlogmcap.groupby("qulogmcap").mean()

"""
Last day mcap factor ENDE
"""




# TRY END

lwkl_ret.to_excel("lwkl_ret.xlsx")
wkl_ret.to_excel("wkl_ret.xlsx")

#print("* lwkl_ret long format table of weekly values of returns of each currency. Also a shifted return of the following week(shift -1)****")
#print(lwkl_ret)

##bring data from wide to long format, i.e., we have three columns: date, cryptocurrency ID (name)
##and value (e.g., price)
prices = pd.melt(prices, id_vars=["date"], value_vars=prices.columns[1:], var_name="cur", value_name="price")
supply = pd.melt(supply, id_vars=["date"], value_vars=supply.columns[1:], var_name="cur", value_name="supply")
mcap = pd.melt(mcap, id_vars=["date"], value_vars=mcap.columns[1:], var_name="cur", value_name="mcap")

#prices["date"].dtypes
##prices["pricewins"] = prices.groupby("cur")["price"].transform(lambda x: winsorize(x, limits=(0.01,0.01), nan_policy="omit"))
##in the following, we will compute some returns that are useful for the seminar paper
##first, we compute daily returns
##get the price from the previous day for each currency
##see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.shift.html
prices["pricey"] = prices.sort_values("date", ascending=True).groupby("cur")["price"].shift(periods=1)
##compute daily returns
prices["ret"] = (prices["price"] / prices["pricey"]) - 1
##RB:shift price by 7 days
prices["pricex"] = prices.sort_values("date", ascending=True).groupby("cur")["price"].shift(periods=7)
##RB:calculate weekly return
prices["returnw"] = (prices["price"]/prices["pricex"]) -1
##RB:create column where the weekly return of the following week is displayed
prices["returnws"] = prices.sort_values("date", ascending=True).groupby("cur")["returnw"].shift(periods=-7)
##prices["price"]=winsorize(prices["price"], limits=[0.01,0.01],nan_policy="omit")
#Try STD:
    
##std_price= prices["ret"].resample("W-thu").std() 
# print("std of price")
##print(std_price)  
#print(prices["cur"])
#type(prices["cur"])
#original_prices= prices.pivot(columns=cur2)
#pricesunstacked= prices.unstack(level=-2)

##second, we compute a daily market crypto return
##i.e., we compute the value-weighted mean across all cryptocurrencies' daily returns
##we "add" the market cap to the prices dataframe
##using the merge function (left join, argument "how") by date and cryptocurrency (argument "on")
##see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
prices = pd.merge(prices, mcap, how="left", on=["date", "cur"])
###*****************TRY***************
#prices.set_index("date", inplace=True)
#priceslwkl_ret= pd.merge(prices,lwkl_ret, how="left",right_index=True)
#prices.reset_index()
####***************TRY


#prices.set_index("date", inplace=True)
#wkl_ret= prices["price"].pct_change().resample("W-thu").agg(lambda x: (x+1).prod()-1)

##for each day, we compute the overall mcap across all currencies for days with returns
#prices = prices[~pd.isna(prices["ret"])]
prices["mcapAgg"] = prices.groupby("date")["mcap"].transform("sum")
mcapAggv=prices["mcapAgg"]
##then we weight each currency's daily return by its mcap normalized by market mcap
prices["wret"] = prices["ret"] * prices["mcap"] / prices["mcapAgg"]
##and finally collapse it at the day level
prices["mret"] = prices.groupby("date")["wret"].transform("sum")
print("mretdurchschnitt_protag")
print(prices["mret"].mean())

##print(prices.groupby("date")["mret"].mean())
##define a new df that has only the daily market return 
market = prices[["date", "mret"]].drop_duplicates()
##Vorbereitende Variable für weekly weighted market return
prices["mretww"]=((prices["price"]/prices["pricex"])-1)*prices["mcap"]/prices["mcapAgg"]
##Weekly weighted market return
prices["mretw"]=prices.groupby("date")["mretww"].transform("sum")
print("prices_weeklymarketret")

print(prices["mretw"].mean())
print("#")
prices['date']= pd.to_datetime(prices['date'])
##meanmretw=prices.resample("W-thu", on="date", kind="timestamp")
#print("meanmretw")
#print(meanmretw)
##shifted mcap by 7 days
prices["mcaps"]=prices.sort_values("date", ascending=True).groupby("cur")["mcap"].shift(periods=-7)
##vorvariable
prices["returnweight"]=prices["returnws"] * prices["mcaps"]/prices["mcapAgg"]
##weighted weekly return shifted
prices["returnwsw"]=prices.groupby("date")["returnweight"].transform("sum")
#prices["date"] = pd.to_datetime(prices["date"], dayfirst=False)
#new Mret into wkl_datav2:
wkl_datav2["mcapAg"] = wkl_datav2.groupby("date")["mcl"].transform("sum")
wkl_datav2["weightmret"]=wkl_datav2["wklyret"]*wkl_datav2["mcl"]/wkl_datav2["mcapAg"]
wkl_datav2["mretnew"]=wkl_datav2.groupby("date")["weightmret"].transform("sum")
mretnewv=wkl_datav2["weightmret"].sum()

print("overallmretnew")
print(mretnewv)
mretnewvweeklyavg=wkl_datav2["mretnew"].mean()

print("mretnewvweeklyavg")
print(mretnewvweeklyavg)

wkl_datav3=pd.merge(wkl_datav2,tbilldiv52, how="left", on=["date"])
wkl_datav3["marketexcret"]=wkl_datav3["mretnew"]-wkl_datav3["Bid Yield"]
wkl_datav3["stratexcret"]=wkl_datav3["finwret"]-wkl_datav3["Bid Yield"]/100
portfexcretv=wkl_datav3.groupby("wklyquint")["stratexcret"].sum()
#wkl_datav2["Tbillweekly"]=
#wkl_datav2.to_excel("wkl_datav2.xlsx")

#prices['date']= pd.to_datetime(prices['date'])
##prices['date_only'] = df['date_time_column'].dt.date
##prices_wide = prices.pivot_table(values=["mcapAgg"], index = "date", columns='cur')

##prices_wide_resampmcapAgg=prices_wide.resample("W-Thu", kind="period").mean()
##wkl_data["weeklymean_mcapAgg"]=prices


##failed attempt at using pct change on the long version of the dataframe=prices
#prices["wkl_ret"]= prices.pct_change(freq="W"))
#
# prices["wkl_ret"]=wkl_ret
#prices["wklyquint"]=prices.groupby("cur")["wkl_ret"].apply(lambda x: pd.qcut(x, 5))
#print(prices["wklyquint"])

## groupby + grouper + pct change failed
##grouped_weekly_prices= prices.groupby(['cur', pd.Grouper(key='date', freq='W')])['pricey'].pct_change()
##!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#grouped_weekly_prices


##print(grouped_weekly_prices.head(14))
##grouped_weekly_prices.to_excel("groupedweeklypricesmean.xlsx")
#type(prices.date[0])
#print(prices1.head())
#prices1[index] = prices1.Date.dt.to_period('W')
#print (prices1)
#print(prices.head())

##prices.reset_index().set_index("date")
##prices.price.resample("W").mean
##print(prices["2016-01-02","2016-01-08"])
##prices.price.resample("1W", on="date").mean()

#print(prices.head(14))
#prices.price.resample("1W")

##prices["ret"].describe()
##y is a df that contains for each ret the correct quintile
##Quintilbereich=pd.qcut(prices["ret"],5)
##prices["Quintilbereich"]=Quintilbereich
##print(prices.head)
##pctchange7=prices.groupby("cur")["pricey"].pct_change(periods=7)
##Quintilspalte erstellen, die die returnw und deren jeweiliges Quintil benennt.
prices["quintilwgrouped"]= prices.groupby(['date'])['returnw'].transform(lambda x: pd.qcut(x, 5,labels=False, duplicates='drop'))

prices["quintilw"]=pd.qcut(prices["returnw"],5)
##prices["date2"]=pd.to_datetime(prices["date"])
##prices.set_index(["date"])
##prices["pctchange7"]=prices.groupby("cur")["pricewins"].pct_change(periods=7)

##pctchange7=prices["pricey"].pct_change(periods=7,freq="W")

##prices["pctchange7"]=pctchange7
##show every 7th row of the prices dataframe in df2
##df2 = prices[prices.index % 7 == 0]

##different attempt that is probably invalid to use:
##show every 7th row in df2
##df2=prices[::7]
##df2=prices.iloc[0::7]
##df2["quintilw"]=pd.qcut(prices["returnw"],5)
##print(df2)
##quintil7=pd.qcut(df2["pctchange7"],5)
##df2["quintil7"]=quintil7
##prices["quintil7"]=quintil7
##avg2=prices.groupby("quintil7").mean()
##Erstelle Durchschnitte für den returnw innerhalb eines Quintilw.
avg3=prices.groupby("quintilw").mean()
##Gruppiere nach Quintil und errechne alle Durchschnittswerte
avg4=prices.groupby("quintilwgrouped").mean()


##print(avg4.iloc[0]['returnws']*(-1)+avg4.iloc[4]["returnws"])
#print("5-1 Result")

##print(df2['returnws'].corr(df2['mretw']))
##print(df2['returnwsw'].corr(df2['mretw']))
##print("Pearson Correlation")
##print(prices)
##avg3=prices.groupby("quintilw")["returnw"].mean()
##print(avg4)

##df3=df2.dropna()
#print(wkl_datav3.isnull().values.any())

##regression
wkl_datav3["finwret"]=wkl_datav3["finwret"].fillna(0)
wkl_datav3["weightmret"]=wkl_datav3["weightmret"].fillna(0)
#wkl_datav5=wkl_datav3.fillna(0)

X = wkl_datav3.iloc[:,11].values.reshape(-1, 1)
#print(X)
Y = wkl_datav3.iloc[:,13].values.reshape(-1, 1)
#print(Y) 
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()

print("Pearson Corr between finwret + weightmret")
print(wkl_datav3["finwret"].corr(wkl_datav3["weightmret"]))
##df2.to_excel("df2.xlsx")

##print(df2)
##df2.to_excel("df2.xlsx")
##print(prices)
##print(prices["returnw"].describe())
##avg3.to_excel("avg3.xlsx")
##print(avg2)
##avg2.to_excel("avg2.xlsx")

##gr = prices.groupby(pd.qcut(prices.pctchange7, 5, labels=false))
##prices["gr"]=gr

##avg4.to_excel("avg4.xlsx")
print("finished avg4")
##print(df2.head)
##df2.to_excel("Testdf2.xlsx")
##prices.to_excel("Prices1.xlsx")
print("finished prices to excel")
##print(prices)

##y.to_excel("Quintile.xlsx")

##x= prices[(prices["ret"] > (prices["ret"].quantile(0.9999)))]
##print(x)
##print(len(x.index))
##print(prices)
##prices.to_excel("DatenRET99Quantil.xlsx")
##output = prices.resample('W').agg({"ret": "sum"}, 
  ##                             loffset = pd.offsets.timedelta(days=-6))
##pd.date_range(start, periods=1000, freq="W")
##prices.info()

##plotting values
##prices.to_excel("Daten11.xlsx")
#plt.scatter(X, Y)


##functions that might be useful for the seminar thesis
##select observations (e.g., at the day level) into percentiles using the qcut function
##see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html
##the calendar package is useful to group dates into weeks, e.g., for week by week momentum
##see https://docs.python.org/3/library/calendar.html

