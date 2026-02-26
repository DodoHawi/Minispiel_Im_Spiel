import ast
from matplotlib.pyplot import title
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots   
from sklearn.preprocessing import MultiLabelBinarizer  


# ---------------------------------------------------------------------------------------------------
# --- Geforderte Skills ---
# ---------------------------------------------------------------------------------------------------
def_skills = pd.read_csv('codebook_database - database.csv', sep=',')

def_skills["Skills_list"] = (
    def_skills["Geforderte(r) Skill(s)"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .str.split(",")
    .apply(lambda xs: [x.strip().title() for x in xs if x.strip() != ""])
)

mlb_skills = MultiLabelBinarizer()
skills_dummies = pd.DataFrame(
    mlb_skills.fit_transform(def_skills["Skills_list"]),
    columns=mlb_skills.classes_,
    index=def_skills.index
)

df_sk = pd.concat([def_skills, skills_dummies], axis=1)
# -> Für jeden Skill eine Spalte mit 1/0 ob vorhanden oder nicht

skills_count = skills_dummies.sum().sort_values(ascending=False)
skills_share = (skills_dummies.mean().sort_values(ascending=False))


fig_skills = px.bar(
    skills_share,
    x=skills_share.values,
    y=skills_share.index,
    orientation='h',
    title='Häufigkeit der geforderten Skills in Minispielen',
    labels={'x': 'Anteil', 'y': 'Skill'},
    color_discrete_sequence=['#87CEFA']
)   

fig_skills.update_layout(font=dict(size=15))
fig_skills.show()


# ---------------------------------------------------------------------------------------------------
# --- Skills - Vergleich der Genres: Heatmap ---
# ---------------------------------------------------------------------------------------------------

skills_genre = (
    df_sk
    .groupby("Genre")[skills_dummies.columns]
    .mean()
)


skills_genre_long = (
    skills_genre
    .reset_index()
    .melt(
        id_vars="Genre",
        var_name="Skill",
        value_name="Anteil"
    )
)

fig_skills_genre = px.density_heatmap(
    skills_genre_long,
    x="Genre",
    y="Skill",
    z="Anteil",
    color_continuous_scale="Blues",
    labels={
        "Genre": "Genre",
        "Skill": "Skill",
        "Anteil": "Anteil der Minispiele"
    },
    title="Verteilung der geforderten Skills auf Genres"
)

fig_skills_genre.update_xaxes(tickangle=45)
fig_skills_genre.update_coloraxes(colorbar_tickformat=".0%", colorbar_title="Anteil der Minispiele")
fig_skills_genre.update_layout(font=dict(size=25))

#fig_skills_genre.show()