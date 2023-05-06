import pandas as pd
import numpy as np

BoM = ['ABC:2', 'XYZ:1', 'IJK:1', 'ABC:1', 'IJK:1', 'XYZ:2', 'DEF:2']
Disti = ['XYZ:2', 'GEF:2', 'ABC:4', 'IJK:2']
L = lambda s: [s.split(':')[0], int(s.split(':')[1])]

BoM= [L(s)for s in BoM]
Disti = [L(s) for s in Disti]

df = pd.DataFrame(BoM,columns=['Part Number','Quantity'])
df1 = pd.DataFrame(Disti,columns=['Part Number','Quantity'])
print(df)
print(df1)

### Get Sequence (order) for df1

s = df['Part Number'].drop_duplicates()
s = s[s.isin(df1['Part Number'])]

s1= df1[~df1['Part Number'].isin(s)]['Part Number']
keeporder = pd.concat([s, s1])

df1 = df1.iloc[list(map(df1['Part Number'].tolist().index, keeporder))].set_index('Part Number').reset_index()


### Get df1 target with respect to df
df1TargetWrt_df = df[['Part Number','Quantity']].merge(df1[['Part Number']],how='right')
df1TargetWrt_df['Quantity'] = df1TargetWrt_df['Quantity'].fillna(0).astype('Int64')

cols = ['Part Number','Quantity']
quantities = df1TargetWrt_df[cols].groupby('Part Number') \
                 .apply(lambda x: x.set_index(['Part Number']).to_dict('list')) \
                 .to_dict()
key_order = df1TargetWrt_df['Part Number'].drop_duplicates()
targetPNQuantities = {}
for k in key_order:
    targetPNQuantities[k] = quantities[k]
print("targetPNQuantities", targetPNQuantities)

### Get indexes for df1 quantities of each PartNumber to add using groupby on those Indexes


def make_groups(Quantities, getPartNumber, targetDict):
    targets = targetDict[getPartNumber]['Quantity']

    if not targets[0]:
        result = np.zeros(len(Quantities), dtype = int)
        return result
    
    result = np.empty(len(Quantities),dtype=np.uint64)

    total = 0
    group = 0

    for i,x_i in enumerate(Quantities):
        total += x_i
        if total > targets[0]:
            targets= targets[1:]
            group += 1
            total = 1
        
        result[i] = group
    
    return result.tolist()

df1 = df1.reindex(df1.index.repeat(df1['Quantity'])).assign(Quantity=1)

g = df1.groupby("Part Number")["Part Number", "Quantity"].apply( lambda x: make_groups(pd.DataFrame(x)["Quantity"].to_numpy(), pd.DataFrame(x)["Part Number"].to_numpy()[0], targetPNQuantities))

h =pd.DataFrame(g).reset_index()
h.columns = df.columns[:1].tolist() +['Quantity']
h = h.explode('Quantity')
index = h.index
h = pd.merge(keeporder, h,
         how='left', on='Part Number',
         sort=False
         ).set_index(index)
print("h['Quantity']", h['Quantity'])

print((df1.groupby(by=["Part Number", h['Quantity']])
        .sum()))


df1targetresults= (df1.groupby(by=["Part Number", h['Quantity']])
        .sum()
        .reset_index(1, drop=True)
        .reset_index())
print('df1targetresults', df1targetresults)

### Sort those df1target and df

df1target_sort = df1targetresults.rename_axis('MyIdx').sort_values(by = ['Part Number', 'MyIdx'], ascending=[True, True])
dfsort= df.rename_axis('MyIdx').sort_values(by = ['Part Number', 'MyIdx'], ascending=[True, True])

print('df1target_sort', df1target_sort)
### Get Matching indexes from df1 (df1target_sort) when compared with df (dfsort)

def getMatchingIndexfrom(df1, df):
    compareColumn = []
    compareColumnIndex =[]
    checked = [-1]
    for index, row in df.iterrows():
        df1Row = df1[df1["Part Number"] == row["Part Number"]]
        if df1Row.shape[0] == 0:
            compareColumn.append("Part Number and Quantity not available")
            compareColumnIndex.append(-1)
        else:
            check = False
            for index1, row1 in df1Row.iterrows():
                if index1 not in checked and row1["Quantity"] == row["Quantity"]:
                    checked.append(index1)
                    compareColumn.append(f"Both matching at index {index1}")
                    compareColumnIndex.append(index1)
                    check = True
                    break
            if check == False:
                compareColumn.append(f"Quantity not matching at index {index1}")
                compareColumnIndex.append(index1)
    df["compare"] = compareColumn
    df["compareIndex"] = compareColumnIndex
    df.sort_values(by=['MyIdx'], inplace= True)
    return df

df = getMatchingIndexfrom(df1target_sort,dfsort)
print('df', df)

### Merging df with df1target_sort by indexes of df1target_sort to get Final Target

final = df.drop('compare', axis=1).merge(df1target_sort, left_on='compareIndex', right_on='MyIdx', how= 'outer').drop('compareIndex', axis=1)
final["Error Flag"]= np.where(final['Quantity_x']==final['Quantity_y'], "", "x")
final['Quantity_x'] = final['Quantity_x'].fillna(-1).astype('Int64').astype(str)
final['Quantity_y'] = final['Quantity_y'].fillna(-1).astype('Int64').astype(str)
final=final.fillna('').replace('-1', ' ')
final.columns = ["BoM PN", "BoM Qty", "Disti PN", "Disti Qty", "Error Flag"]
print('final', final)

unifiedDict =final.to_dict()
print('UnifiedDict', unifiedDict)