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

# Ergänzen des absoluten Anteils "Nein"
feature_summary["Nein"] = feature_summary["Gesamt"] - feature_summary["Ja"]
feature_summary["Ja_pct"] = (feature_summary["Ja"] / feature_summary["Gesamt"]).round(2)
feature_summary["Nein_pct"] = (feature_summary["Nein"] / feature_summary["Gesamt"]).round(2)


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

fig1.show()