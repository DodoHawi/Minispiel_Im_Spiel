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

mot = ["Geschicklichkeit", "Timing", "Präzision", "Mashing", "Reaktionsgeschwindigkeit"]

skills_df = skills_count.reset_index()
skills_df.columns = ["Skills", "Anzahl"]

skills_df["Kategorie"] = skills_df["Skills"].apply(lambda x: "Motorische Skills" if x in mot else "Nicht motorische Skills")


total_minigames = len(df_sk)
skills_df["Prozent"] = skills_df["Anzahl"] / total_minigames
skills_df["Label"] = (
    skills_df["Anzahl"].astype(str)
    + " (" +
    (skills_df["Prozent"]*100).round(1).astype(str)
    + "%)"
)

fig_skills = px.bar(
    skills_df,
    x="Anzahl",
    y="Skills",
    color = "Kategorie",
    text="Label",
    orientation='h',
    title='Häufigkeit der geforderten Skills in Minispielen',
    labels={'x': 'Anzahl', 'y': 'Skill'},
    color_discrete_map={
        "Motorische Skills": "#87CEFA",
        "Nicht motorische Skills": "#FF6A6A"
    }
)   

fig_skills.update_layout(font=dict(size=20))
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