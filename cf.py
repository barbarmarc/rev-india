import pandas as pd

regions = ['NR', 'WR', 'SR', 'ER', 'NER']
tech = ['w0', 'w1', 'w2', 'g126_84', 'g126_102', 's']

for r in regions:
    for t in tech:
        df_mean = []
        df = pd.read_csv('india_points/'+r+'_'+t+'_cf_profile.csv', index_col=0)
        for index, row in df.iterrows():
            df_mean.append(row.mean())
        df_mean = pd.DataFrame(df_mean)
        df_mean.to_csv('india_points/mean/'+r+'_'+t+'_cf.csv')
