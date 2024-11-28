import pandas as pd

df4 = pd.read_csv("./vectorized_for_app.csv")
print(df4.Lyric_Embedding[0].shape)
