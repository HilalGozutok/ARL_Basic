         ############################################
             # ASSOCIATION_RULE_LEARNING_BASIC #
         ############################################

############################################
# Data Preparing
############################################

# pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
from helpers.helpers import create_invoice_product_df

#The process of reading the data set.

df_ = pd.read_excel(r"C:\Users\LENOVO\PycharmProjects\DSMLBC4\datasets\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()

from helpers.helpers import check_df
check_df(df)

from helpers.helpers import crm_data_prep

df = crm_data_prep(df)
check_df(df)

#Country selection -> [Germany]
df_ger = df[df['Country'] == "Germany"]
check_df(df_ger)

df_ger.groupby(['Invoice', 'StockCode']).agg({"Quantity": "sum"}).head(100)
# çıktıda ürünün ilk bulunduğu fatura [0:6, 0:12] de görüldüğü için aralık böyle şeçilmiştir.
df_ger.groupby(['Invoice', 'StockCode']).agg({"Quantity": "sum"}).unstack().iloc[0:6, 0:12]

# Control
df[(df["StockCode"] == 16016) & (df["Invoice"] == 536983)]


df_ger.groupby(['Invoice', 'StockCode']).\
    agg({"Quantity": "sum"}).\
    unstack().fillna(0).iloc[0:6, 0:12]

# Apply, satır ve sütuna göre(axis özelinde seçim yapılır) applymap ise tüm elemanlara göre çalışır.
df_ger.groupby(['Invoice', 'StockCode']).\
    agg({"Quantity": "sum"}).\
    unstack().fillna(0).\
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:6, 0:12]


# fillna(0) fills NA values with 0.
ger_inv_pro_df = create_invoice_product_df(df_ger)

ger_inv_pro_df.head()


# How many unique products are in each invoice?
new_df_ger = df_ger.groupby(['Invoice', 'Description']).agg({'Quantity': "sum"}).fillna(0).\
        applymap(lambda x: 1 if x > 0 else 0)
new_df_ger.head(30)
new_df_ger.reset_index(inplace=True)
new_df_ger.head(30)
new_df_ger.groupby('Invoice').agg({'Quantity': "sum"})

# How many unique baskets are each product in?
df_ger.groupby("Description").agg({"Invoice":"nunique"})


############################################
# Creation Of Association Rules
############################################

frequent_itemsets = apriori(ger_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()
rules.sort_values("lift", ascending=False).head()

#leverage => the support value gives priority to high values.
#conviction; x and y are two products. Denotes the expected frequency of x without y.

############################################
# Code Functionalization
############################################
import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules
from helpers.helpers import crm_data_prep,create_invoice_product_df


df_ = pd.read_excel(r"C:\Users\LENOVO\PycharmProjects\DSMLBC4\datasets\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df = crm_data_prep(df)

def create_rules(dataframe, country=False, head=5):
    if country:
        dataframe = dataframe[dataframe['Country'] == country]
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))
    else:
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))
    return rules

rules = create_rules(df)







