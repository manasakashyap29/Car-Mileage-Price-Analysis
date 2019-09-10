import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

US_INFLATION_AVG_2001_2019 = 2.04

def preprocess(df):
    #Transpose data
    df = df.transpose()

    #Fix headers
    df.columns = df.iloc[0]
    df = df.rename(columns={np.NaN : "Model", "MSRP": "MSRP (USD)"})
    df = df[1:]

    # Remove specs from Model
    df["Model"] = df["Model"].replace(to_replace =' Specs:.*', value = '', regex = True)

    #Fix Make names
    df["Model"] = df["Model"].replace(to_replace ='Alfa Romeo', value = 'Alfa-Romeo', regex = True)
    df["Model"] = df["Model"].replace(to_replace ='Aston Martin', value = 'Aston-Martin', regex = True)
    df["Model"] = df["Model"].replace(to_replace ='Land Rover', value = 'Land-Rover', regex = True)

    df = df.dropna()

    #Convert prices to integers
    df["MSRP (USD)"] = df["MSRP (USD)"].replace(to_replace =['\$', ','], value = '', regex = True).astype(float)


    #Split Year, Make and Model
    model_split = df['Model'].str.split(" ", n=2, expand=True)
    df["Year"] = model_split[0].astype(int)
    df["Make"] = model_split[1]
    df["Model"] = model_split[2]

    #Discover incorrect splits, fix above
    #print(df.Make.unique())
    #print(df[df.Model.str.contains(" ")].Model.unique())

    #Split Highway and city mileage
    mileage_split = df['Gas Mileage'].str.split("/", n=1, expand=True)
    df["City mpg"] = mileage_split[0].replace(to_replace=' mpg City', value='', regex=True).astype(float)
    df["Highway mpg"] = mileage_split[1].replace(to_replace=' mpg Hwy', value='', regex=True).astype(float)
    df = df.drop('Gas Mileage', 1)

    df = df.dropna()

    #df = df.set_index(['Make', 'Model', 'Year'])
    df = df.groupby(['Make', 'Model', 'Year']).mean()

    return df

def analyse_mpg_growth(mpg, type_mpg):
    mpg = mpg.reset_index()
    mpg_max = mpg.loc[mpg.groupby(['Make', 'Model'])['Year'].idxmax()]
    mpg_min = mpg.loc[mpg.groupby(['Make', 'Model'])['Year'].idxmin()]

    mpg_max = mpg_max.set_index(['Make', 'Model'])
    mpg_min = mpg_min.set_index(['Make', 'Model'])

    mpg_merged = mpg_max.join(mpg_min, lsuffix='_maxY', rsuffix='_minY')
    
    mpg_merged["TimeDiff"] = mpg_merged["Year_maxY"] - mpg_merged["Year_minY"]
    
    #AEI: Annualized average efficiency improvement
    mpg_merged["AEI"] = (np.expm1((np.log(mpg_merged[type_mpg+" mpg_maxY"]/mpg_merged[type_mpg+" mpg_minY"]))/mpg_merged["TimeDiff"]))*100
    mpg_merged = mpg_merged.dropna()
    mpg_merged = mpg_merged.reset_index()
    mpg_merged = mpg_merged[['Make', 'AEI']]
    mpg_merged = mpg_merged.groupby('Make').mean()

    plot = mpg_merged.plot.bar()

    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig(type_mpg+"_aei.png")


def analyse_price_growth(avg_prices):
    avg_prices = avg_prices.reset_index()
    avg_prices_max = avg_prices.loc[avg_prices.groupby(['Make', 'Model'])['Year'].idxmax()]
    avg_prices_min = avg_prices.loc[avg_prices.groupby(['Make', 'Model'])['Year'].idxmin()]

    avg_prices_max = avg_prices_max.set_index(['Make', 'Model'])
    avg_prices_min = avg_prices_min.set_index(['Make', 'Model'])

    avg_prices_merged = avg_prices_max.join(avg_prices_min, lsuffix='_maxY', rsuffix='_minY')
    
    avg_prices_merged["TimeDiff"] = avg_prices_merged["Year_maxY"] - avg_prices_merged["Year_minY"]
    
    #API: Annualized Average Price Increase
    avg_prices_merged["API"] = (np.expm1((np.log(avg_prices_merged["MSRP (USD)_maxY"]/avg_prices_merged["MSRP (USD)_minY"]))/avg_prices_merged["TimeDiff"]))*100
    avg_prices_merged = avg_prices_merged.dropna()
    avg_prices_merged = avg_prices_merged.reset_index()
    avg_prices_merged = avg_prices_merged[['Make', 'API']]
    avg_prices_merged = avg_prices_merged.groupby('Make').mean()
    avg_prices_merged["Avg. Inflation"] = US_INFLATION_AVG_2001_2019

    ax = avg_prices_merged[["API"]].plot.bar()
    ax = avg_prices_merged[["Avg. Inflation"]].plot(linestyle='-', color='red', ax=ax,rot=90)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig("api.png")


if __name__ == '__main__':
    #Extract models, price and mileage
    df = pd.read_csv('fullspecs.csv',sep=',', header = None, nrows=3)
    df = preprocess(df)

    #Analyse Annualized Efficiency Improvements for different car-makers
    city = df["City mpg"]
    highway = df["Highway mpg"]

    city_aei = analyse_mpg_growth(city, "City")
    highway_aei = analyse_mpg_growth(highway, "Highway")

    #Analyse Annualized Average car price increase, compared to inflation
    avg_prices = df["MSRP (USD)"]
    analyse_price_growth(avg_prices)
