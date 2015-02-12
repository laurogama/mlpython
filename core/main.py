import sys

import matplotlib


matplotlib.use('Qt4Agg')
import psycopg2
import pandas as pd
import matplotlib.pylab as plt
# import matplotlib.rcsetup as rcsetup

# print(rcsetup.all_backends)
plt.matplotlib.matplotlib_fname()


def connect_db():
    global con, cur, ver, e
    con = None
    try:

        con = psycopg2.connect(database='machine', user='postgres', password='arete123')
        cur = con.cursor()
        cur.execute('SELECT version()')
        ver = cur.fetchone()
        print ver


    except psycopg2.DatabaseError, e:
        print 'Error %s' % e
        sys.exit(1)


    finally:

        if con:
            con.close()


def plotting():
    # plt.ion()
    countries = ['France', 'Spain', 'Sweden', 'Germany', 'Finland', 'Poland', 'Italy',
                 'United Kingdom', 'Romania', 'Greece', 'Bulgaria', 'Hungary',
                 'Portugal', 'Austria', 'Czech Republic', 'Ireland', 'Lithuania', 'Latvia',
                 'Croatia', 'Slovakia', 'Estonia', 'Denmark', 'Netherlands', 'Belgium']
    extensions = [547030, 504782, 450295, 357022, 338145, 312685, 301340, 243610, 238391,
                  131940, 110879, 93028, 92090, 83871, 78867, 70273, 65300, 64589, 56594,
                  49035, 45228, 43094, 41543, 30528]
    populations = [63.8, 47, 9.55, 81.8, 5.42, 38.3, 61.1, 63.2, 21.3, 11.4, 7.35,
                   9.93, 10.7, 8.44, 10.6, 4.63, 3.28, 2.23, 4.38, 5.49, 1.34, 5.61,
                   16.8, 10.8]
    life_expectancies = [81.8, 82.1, 81.8, 80.7, 80.5, 76.4, 82.4, 80.5, 73.8, 80.8, 73.5,
                         74.6, 79.9, 81.1, 77.7, 80.7, 72.1, 72.2, 77, 75.4, 74.4, 79.4, 81, 80.5]
    data = {'extension': pd.Series(extensions, index=countries),
            'population': pd.Series(populations, index=countries),
            'life expectancy': pd.Series(life_expectancies, index=countries)}

    df = pd.DataFrame(data)
    print(df)
    df = df.sort('life expectancy')
    fig, axes = plt.subplots(nrows=3, ncols=1)
    for i, c in enumerate(df.columns):
        df[c].plot(kind='bar', ax=axes[i], figsize=(12, 10), title=c)
    plt.show()
    # plt.savefig('../output/EU1.png', bbox_inches='tight')


def satellite(country):
    df = pd.read_excel("../datasets/UCS_Satellite_Database_8-1-14.xls", 0)
    unique_countries = df['Country of Operator/Owner'].unique()

    a = 0
    for x in unique_countries:
        if country in x:
            a = a + 1
            print x
    print(a)


def group_country():
    df = pd.read_excel("../datasets/UCS_Satellite_Database_8-1-14.xls", 'Sheet1', index_col=None)
    # gb = df['Country of Operator/Owner', 'Date of Launch']
    # plt.plot(gb.groupby('Country of Operator/Owner'))
    # plt.show()
    print(df.head())


def consume_wallmart_data():
    df = pd.read_csv("../datasets/kaggle/wallmart/features.csv", na_values=['NaN'])
    print(df.head())
    print (df.columns)
    # fig, axes = plt.subplots(nrows=12, ncols=1)
    # data = pd.Series(df['Date'])
    # print(data)
    df.plot(kind='bar')
    plt.show()
    # for i, c in enumerate(df.columns):
    # df[c].plot(kind='line', ax=axes[i], figsize=(12, 10), title=c)
    # plt.show()


if __name__ == "__main__":
    # plotting()
    # group_country()
    consume_wallmart_data()
    # satellite("USA")