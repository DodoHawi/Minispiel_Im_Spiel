import ast
from matplotlib.pyplot import title
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots   
from sklearn.preprocessing import MultiLabelBinarizer  

# --- Komplexität-Plot ---
# Neue CSV einlesen
df_komp = pd.read_csv('codebook_database - database.csv', sep=',')

# Whitespace in Spaltennamen entfernen
df_komp.columns = df_komp.columns.str.strip()

# Absolute Anzahl jeder Komplexität insgesamt
komp_counts = df_komp['Komplexität'].value_counts().reindex(['hoch', 'mittel', 'gering']).reset_index()
komp_counts.columns = ['Komplexität', 'Anteil']

# Plot: Verteilung der Komplexität (Pie Chart)
fig_komp_count = px.pie(
    komp_counts,
    values='Anteil',
    names='Komplexität',
    title='Verteilung der Komplexität-Level',
    color='Komplexität',
    color_discrete_map={'gering': '#87CEFA', 'mittel': '#FFD700', 'hoch': '#FF6A6A'}
)
fig_komp_count.update_layout(
    font=dict(size=25), 
    legend=dict(
        title=dict(text = "Komplexität")))

#fig_komp_count.show()

#----------------------------------------------------------------------------------------------------
# Komplexität nach Genre: Gruppieren und Plotten
# ----------------------------------------------------------------------------------------------------

df_komp["Genre_list"] = (
    df_komp["Genre"]
    .fillna("")                      # NaN -> ""
    .astype(str)
    .str.split(",")                  # oder dein Separator
    .apply(lambda xs: [x.strip() for x in xs if x.strip() != ""])
)

df_komp_exploded = df_komp.explode("Genre_list")

# Anzahl pro Genre × Komplexität
komp_by_genre = (
    df_komp_exploded
    .groupby(["Genre_list", "Komplexität"])
    .size()
    .reset_index(name="count")
)


# Anteile innerhalb jedes Genres
komp_by_genre["Anteil"] = (
    komp_by_genre["count"]
    / komp_by_genre.groupby("Genre_list")["count"].transform("sum")
)


fig_komp_genre = px.bar(
    komp_by_genre,
    x="Genre_list",
    y="Anteil",
    color="Komplexität",
    title="Komplexität der Minispiele nach Genre",
    category_orders={
        "Komplexität": ["gering", "mittel", "hoch"]
    },
    color_discrete_map={
        "gering": "#87CEFA",
        "mittel": "#FFD700",
        "hoch": "#FF6A6A"
    }
)

fig_komp_genre.update_yaxes(tickformat=".0%")
fig_komp_genre.update_xaxes(tickangle=45)
fig_komp_genre.update_layout(
    legend_title_text="Komplexität",
    font=dict(size=22)
)

fig_komp_genre.show()