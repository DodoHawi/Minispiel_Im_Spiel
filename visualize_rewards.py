import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots   
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
belohnung_share = belohnung_dummies.mean().sort_values(ascending=False)

fig_belohnung = px.bar(
    belohnung_share,
    x=belohnung_share.values,
    y=belohnung_share.index,
    orientation='h',
    title='Häufigkeit der Belohnungen in Minispielen',
    labels={'x': 'Anteil', 'y': 'Belohnungstyp'},
    color_discrete_sequence=['#87CEFA']
)   

#fig_belohnung.update_layout(font=dict(size=25))

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