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
balanced_df["Label"] = balanced_df["Label"].map({"ham":0, "spam":1})

def random_split(df,train_frac,validation_frac):
    df = df.sample(
        frac=1,random_state=123
    ).reset_index(drop=True)
    train_end = int(len(df)*train_frac)
    validation_end = train_end + int(len(df)*validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df,validation_df,test_df

train_df, validation_df, test_df = random_split(balanced_df,train_frac=0.7,validation_frac=0.2)
train_df.to_csv("train.csv",index=None)
validation_df.to_csv("validation.csv",index=None)
test_df.to_csv("test.csv",index=None)