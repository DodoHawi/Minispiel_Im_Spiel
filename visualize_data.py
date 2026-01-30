from matplotlib.pyplot import title
import pandas as pd
import plotly.express as px
import numpy as np
#import matplotlib.pyplot as plt

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
.rename(columns={"present": "Anteil_Ja", "feature": "Kategorie"})
)

# Ergänzen des Anteils "Nein"
feature_summary["Anteil_Nein"] = 1 - feature_summary["Anteil_Ja"]

# Reshape für Plotting
feature_summary_melted = feature_summary.melt(
    id_vars=["Kategorie"],
    value_vars=["Anteil_Ja", "Anteil_Nein"],
    var_name="Ergebnis",
    value_name="Anteil"
)

# Vergleich nach Genre
genre_share = (
    df_long.groupby(["Genre", "feature"])["present"]
    .mean()
    .reset_index(name="Anteil")
)

# Plot der Ergebnisse
fig1 = px.bar(
    feature_summary_melted,
    x="Anteil",
    y="Kategorie",
    color="Ergebnis",
    orientation='h',
    title="Ergebnis pro Kategorie",
    labels={"Anteil": "Anteil", "Kategorie": "Kategorie", "Ergebnis": "Ergebnis"},
    color_discrete_map={"Anteil_Ja": "#87CEFA", "Anteil_Nein": "#FF6A6A"}
)

fig2 = px.bar(
    genre_share,
    x="Anteil",
    y="feature",
    color="Genre",
    orientation='h',
    barmode='group',
    title="Anteil 'Ja' pro Kategorie und Genre",
)

fig3 = px.bar(
    genre_share,
    x="Genre",
    y="Anteil", 
    facet_col="feature",
    facet_col_wrap=3,
    labels={"Anteil": "Anteil 'Ja'", "Genre": "Genre", "feature": "Kategorie"},
    title="Anteil 'Ja' pro Kategorie und Genre (Facettenansicht)"
)

#fig1.show()
#fig2.show()
fig3.show()

#Plot der Ergebnisse mit Matplotlib
#plt.figure(figsize=(10, 6))
#plt.barh(feature_summary['Kategorie'], feature_summary['proportion_yes'], color='skyblue')
#plt.xlabel('Proportion of "Ja"')
#plt.ylabel('Kategorie')
#plt.title('Proportion of "Ja" Responses per Kategorie')
#plt.tight_layout()
#plt.show()


#print(feature_summary)