import pandas as pd
import plotly.express as px

df = pd.read_csv('minispiele_raw_clean.csv', sep=';')

# Spalten mit nur Ja/Nein/NaN
binary_columns = [
    col for col in df.columns
    if df[col].dropna().isin(['Ja', 'Nein']).all()
]

# Ja/Nein in 1/0 umwandeln
df_bin = df.copy()
for col in binary_columns:
    df_bin[col] = df_bin[col].map({'Ja': 1, 'Nein': 0}) 
    
# Wide to long format
df_long = df_bin.melt(
    id_vars=["Spiel", "Genre", "Minispiel"],
    value_vars=binary_columns,
    var_name="feature",
    value_name="present"
)

# Anteil "Ja" pro Feature berechnen
feature_summary = (
    df_long.groupby("feature")["present"]
    .mean()
    .reset_index()
    .rename(columns={"present": "proportion_yes"})
)

# Plot der Ergebnisse

print(df_long)
print(feature_summary)