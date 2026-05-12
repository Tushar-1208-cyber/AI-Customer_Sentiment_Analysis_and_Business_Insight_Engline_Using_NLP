import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    
    print("Dataset Loaded ✅")
    print(df.head())
    print(df.columns)
    
    return df

if __name__ == "__main__":
    df = load_dataset("data/cleaned_reviews.csv")
    print(df['sentiment'].value_counts())