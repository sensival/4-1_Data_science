import pandas as pd
import itertools
df = pd.read_csv('marketbasket.csv', index_col=0)

def support(df, item_lst):
    return (df[list(item_lst)].sum(axis=1)==len(item_lst)).mean()

def make_all_set_over_support(df, support_threshold):
    items = []
    single_items = [col for col in df.columns if support(df, [col]) > support_threshold] # size 1 items
    
    size = 2
    while True:
        new_items = []
        for item_cand in itertools.combinations(single_items, size):
            #print(item_cand, (df[list(item_cand)].sum(axis=1)==size).mean())
            if support(df, list(item_cand)) > support_threshold:
                new_items.append(list(item_cand))
        if len(new_items)==0:
            break
        else:
            items+=new_items
            size+=1
    items += [ [s] for s in single_items]# 이렇게 해줘야 모든 type이 list가 됨
    return items

def make_confidence_lst(df, item_set_over_support, confidence_threshold):
    r_lst = []
    for item1 in item_set_over_support:
        for item2 in item_set_over_support:
            if len(set(item1).intersection(set(item2)))==0:
                conf = support(df, list(set(item1).union(set(item2))))/ support(df, item1)
                if conf > confidence_threshold:
                    r_lst.append((item1, item2, conf))
            else:
                continue
    return sorted(r_lst, key=lambda x: x[2], reverse=True)

def make_lift_lst(df, item_set_over_support, lift_threhsold):
    r_lst = []
    for item1 in item_set_over_support:
        for item2 in item_set_over_support:
            if len(set(item1).intersection(set(item2)))==0:
                lift = support(df, list(set(item1).union(set(item2))))
                lift /= support(df, item1)
                lift /= support(df, item2)
                if lift > lift_threhsold:
                    r_lst.append((item1, item2, lift))
            else:
                continue
    return sorted(r_lst, key=lambda x: x[2], reverse=True)

over_support_lst = make_all_set_over_support(df, 0.07)# 0.05로 하면 두 개짜리도 나옴. 로 하면 3개 짜리도 나옴
print("over support list")
print(over_support_lst)
print("-----------------")
print("over confidence list")
for a, b, conf in  make_confidence_lst(df, over_support_lst, 0.53):
    print("{} => {}: {}".format(a, b, conf))
print("-----------------")
print("over lift list")
for a, b, lift in  make_lift_lst(df, over_support_lst, 5.6):
    print("{} => {}: {}".format(a, b, lift))
print("-----------------")
