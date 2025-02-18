import pandas as pd

data_file_path = "sms_spam_collection/SMSSpamCollection.tsv"
df = pd.read_csv(
    data_file_path,sep="\t",header=None,names=["Label","Text"]
)
print(df["Label"].value_counts())

def create_balanced_dataset(df):
    num_spam = df[df["Label"]=="spam"].shape[0]
    ham_subset = df[df["Label"]=="ham"].sample(
        num_spam,random_state=123
    )
    balanced_df = pd.concat([ham_subset,df[df["Label"]=="spam"]])
    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())