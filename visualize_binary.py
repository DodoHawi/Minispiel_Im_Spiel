
import pandas as pd
import plotly.express as px



df = pd.read_csv('codebook_database - database.csv', sep=',')

# Spalten mit nur Ja/Nein/NaN
binary_columns = [
    col for col in df.columns
    if df[col].dropna().isin(['ja', 'nein']).all()
]

# Ja/Nein in 1/0 umwandeln
df_bin = df.copy()
for col in binary_columns:
    df_bin[col] = df_bin[col].map({'ja': 1, 'nein': 0}) 
    
# Wide to long format
df_long = df_bin.melt(
    id_vars=["Spielname", "Genre", "Minispiel"],
    value_vars=binary_columns,
    var_name="feature",
    value_name="present"
)

# Anteil "Ja" pro Feature berechnen
#feature_summary = (
#    df_long.groupby("feature")["present"]
#    .mean()
#    .reset_index()
#.rename(columns={"present": "Anteil_Ja", "feature": "Kategorie"})
#)

# Absolute Zahlen pro Feature berechnen
feature_summary = (
    df_long.groupby("feature")["present"]
    .agg(
        Ja="sum",   #Anzahl "Ja"
        Gesamt="count" #Gesamtanzahl
    )
    .reset_index()
    .rename(columns={"feature": "Kategorie"})
)

# Ergänzen des Anteils "Nein"
#feature_summary["Anteil_Nein"] = 1 - feature_summary["Anteil_Ja"]

# Ergänzen des absoluten Anteils "Nein"
feature_summary["Nein"] = feature_summary["Gesamt"] - feature_summary["Ja"]
feature_summary["Ja_pct"] = (feature_summary["Ja"] / feature_summary["Gesamt"]).round(2)
feature_summary["Nein_pct"] = (feature_summary["Nein"] / feature_summary["Gesamt"]).round(2)


# Reshape für Plotting
#feature_summary_melted = feature_summary.melt(
#    id_vars=["Kategorie"],
#    value_vars=["Anteil_Ja", "Anteil_Nein"],
#    var_name="Ergebnis",
#    value_name="Anteil"
#)

feature_summary_melted = feature_summary.melt(
    id_vars=["Kategorie", "Gesamt"],
    value_vars=["Ja", "Nein"],
    var_name="Ergebnis",
    value_name="Anzahl"
)

feature_summary_melted["Prozent"] = feature_summary_melted.apply(
    lambda row: row["Anzahl"] / row["Gesamt"],
    axis=1
)

feature_summary_melted["Label"] = (
    feature_summary_melted["Anzahl"].astype(str)
    + " ("
    + (feature_summary_melted["Prozent"] * 100).round(1).astype(str)
    + "%)"
)

#feature_summary_melted["Anteil"] = feature_summary_melted["Anteil"].round(2)

# Vergleich nach Genre
genre_share = (
    df_long.groupby(["Genre", "feature"])["present"]
    .mean()
    .reset_index(name="Anzahl")
)

# Plot der Ergebnisse
fig1 = px.bar(
    feature_summary_melted,
    x="Anzahl",
    y="Kategorie",
    text="Label",
    color="Ergebnis",
    orientation='h',
    title="Ergebnis pro Kategorie",
    labels={"Anzahl": "Anzahl", "Kategorie": "Kategorie", "Ergebnis": "Ergebnis"},
    color_discrete_map={"Ja": "#87CEFA", "Nein": "#FF6A6A"}
)

#fig1.update_layout(
#    font=dict(size=20))


fig2 = px.bar(
    genre_share,
    x="Anzahl",
    y="feature",
    color="Genre",
    orientation='h',
    barmode='group',
    title="Anzahl 'Ja' pro Kategorie und Genre",
)

fig3 = px.bar(
    genre_share,
    x="Genre",
    y="Anzahl", 
    facet_col="feature",
    facet_col_wrap=3,
    labels={"Anzahl": "Anzahl 'Ja'", "Genre": "Genre", "feature": "Kategorie"},
    title="Anzahl 'Ja' pro Kategorie und Genre"
)

fig1.show()
#fig2.show()
#fig3.show()
