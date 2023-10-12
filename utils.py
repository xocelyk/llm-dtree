import pandas as pd

def one_hot_encode(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep=' ')
        df = pd.concat([df, dummies], axis=1)
        df.drop(col, axis=1, inplace=True)
    return df

def balance_data(df):
    # assume we have a label column called 'label' with vals 0 and 1
    min_label = df['label'].value_counts().idxmin()
    min_label_count = df['label'].value_counts()[min_label]
    # sample min_label_count from each label
    df_min = df[df['label'] == min_label]
    df_max = df[df['label'] != min_label].sample(min_label_count)
    df = pd.concat([df_min, df_max])
    df = df.sample(frac=1)
    return df