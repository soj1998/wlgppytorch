import pandas as pd

import gpdata

# 三个字段 name, site, age
nme = ["Google", "Runoob", "Taobao", "Wiki"]
st = ["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]
ag = [90, 40, 80, 90]

# 字典
dict = {'name': nme, 'site': st, 'age': ag}

df = pd.DataFrame(dict)

a = df.loc[df.shape[0]-1, 'name']


c = gpdata.GetData('2529','海源复材','20230101','20231231')
a = c.get_gp_xy()
xylist = a
x = []
y = []
for row in xylist.itertuples():
    if len(row.zuhe) > 0 and len(row.y) > 0:
        ret1 = [int(i) for i in str(row.y).split(' ')]
        x.append(row.zuhe)
        y.append(row.y)
a.to_excel('aaa.xlsx')

print(1)
