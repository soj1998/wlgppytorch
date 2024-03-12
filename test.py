import pandas as pd

data = {'name': ['Tom', 'Jerry'], 'age': [20, 21]}

df = pd.DataFrame(data)

new_data = {'name': 'Mike', 'age': 22}

df._append(new_data, ignore_index=True)

print(round(1.6))