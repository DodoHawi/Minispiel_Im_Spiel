import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MultiLabelBinarizer  

# ---------------------------------------------------------------------------------------------------
# --- Belohnungen ---
# ---------------------------------------------------------------------------------------------------
df_bel =pd.read_csv('codebook_database - database.csv', sep=',')

df_bel["Belohnungen_list"] = (
    df_bel["Belohnungen"]
    .fillna("")                      # NaN -> ""
    .astype(str)
    .str.strip()                        # Whitespace entfernen
    .str.lower()                        # Kleinbuchstaben
    .str.split(",")                  # oder dein Separator
    .apply(lambda xs: [x.strip().title() for x in xs if x.strip() != ""])
)

mlb = MultiLabelBinarizer()
belohnung_dummies = pd.DataFrame(
    mlb.fit_transform(df_bel["Belohnungen_list"]),
    columns=mlb.classes_,
    index=df_bel.index
)

df_belohnung = pd.concat([df_bel, belohnung_dummies], axis=1)
# -> Für jede Belohnung eine Spalte mit 1/0 ob vorhanden oder nicht

belohnung_count = belohnung_dummies.sum().sort_values(ascending=False)

oek = ["Spielwährung", "Verkäufliche/Eintauschbare Items", "Ressourcen"]

belohnung_df = belohnung_count.reset_index()
belohnung_df.columns = ["Belohnung", "Anzahl"]

belohnung_df["Kategorie"] = belohnung_df["Belohnung"].apply(lambda x: "Ökonomische Belohnung" if x in oek else "Nicht ökonomische Belohnung")

total_minigames = len(df_bel)
belohnung_df["Prozent"] = belohnung_df["Anzahl"] / total_minigames
belohnung_df["Label"] = (
    belohnung_df["Anzahl"].astype(str)
    + " (" +
    (belohnung_df["Prozent"]*100).round(1).astype(str)
    + "%)"
)

fig_belohnung = px.bar(
    belohnung_df,
    x="Anzahl",
    y="Belohnung",
    orientation='h',
    color = "Kategorie",
    text = "Label",
    title='Häufigkeit der Belohnungen in Minispielen',
    labels={'x': 'Anzahl', 'y': 'Belohnungstyp'},
    color_discrete_map={
        "Ökonomische Belohnung": "#87CEFA",
        "Nicht ökonomische Belohnung": "#FF6A6A"
    }
)   

fig_belohnung.update_layout(font=dict(size=20))

fig_belohnung.show()

# ---------------------------------------------------------------------------------------------------
# --- Belohnungen - Vergleich der Genres: Heatmap ---
# ---------------------------------------------------------------------------------------------------


belohnung_genre = (
    df_belohnung
    .groupby("Genre")[belohnung_dummies.columns]
    .mean()
)


belohnung_genre_long = (
    belohnung_genre
    .reset_index()
    .melt(
        id_vars="Genre",
        var_name="Belohnung",
        value_name="Anteil"
    )
)

fig_belohnung_genre = px.density_heatmap(
    belohnung_genre_long,
    x="Genre",
    y="Belohnung",
    z="Anteil",
    color_continuous_scale="Blues",
    labels={
        "Genre": "Genre",
        "Belohnung": "Belohnungstyp",
        "Anteil": "Anteil der Minispiele"
    },
    title="Verteilung der Belohnungen auf Genres"
)

fig_belohnung_genre.update_xaxes(tickangle=45)
fig_belohnung_genre.update_coloraxes(colorbar_tickformat=".0%", colorbar_title="Anteil der Minispiele")
#fig_belohnung_genre.update_layout(font=dict(size=25))

#fig_belohnung_genre.show()