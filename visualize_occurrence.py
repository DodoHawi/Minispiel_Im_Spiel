import ast
from matplotlib.pyplot import title
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots   
from sklearn.preprocessing import MultiLabelBinarizer  

# ---------------------------------------------------------------------------------------------------
# --- Auftreten-Plot ---
# ---------------------------------------------------------------------------------------------------

# Neue CSV einlesen
df_auf = pd.read_csv('codebook_database - database.csv', sep=',')

# Whitespace in Spaltennamen entfernen
df_auf.columns = df_auf.columns.str.strip()

# Absolute Anzahl jeder Komplexität insgesamt
auf_counts = df_auf['Auftreten'].value_counts().reindex(['frei','einmalig', 'selten', 'häufig']).reset_index()
auf_counts['Auftreten'] = auf_counts['Auftreten'].replace({
    'frei': 'Frei',
    'einmalig': 'Einmalig',
    'selten': 'Selten',
    'häufig': 'Häufig'
})
auf_counts.columns = ['Auftreten', 'Anteil']

# Plot: Verteilung des Auftretens (Pie Chart)
fig_auf_count = px.pie(
    auf_counts,
    values='Anteil',
    names='Auftreten',
    title='Verteilung der Auftreten-Level',
    color='Auftreten',
    color_discrete_map={'Frei': '#87CEFA', 'Einmalig': '#FFD700', 'Selten': '#FF6A6A', 'Häufig': '#8B4513'}
)
fig_auf_count.update_layout(
    font=dict(size=25), 
    legend=dict(
        title=dict(text = "Auftreten")))

fig_auf_count.show()

#----------------------------------------------------------------------------------------------------
# Auftreten nach Genre
# ----------------------------------------------------------------------------------------------------

# Anzahl pro Genre × Auftreten
auf_by_genre = (
    df_auf
    .groupby(["Genre", "Auftreten"])
    .size()
    .reset_index(name="count")
)

# Anteile innerhalb jedes Genres
auf_by_genre["Anteil"] = (
    auf_by_genre["count"]
    / auf_by_genre.groupby("Genre")["count"].transform("sum")
)


fig_auf_genre = px.bar(
    auf_by_genre,
    x="Genre",
    y="Anteil",
    color="Auftreten",
    title="Auftreten der Minispiele nach Genre",
    category_orders={
        "Auftreten": ["Frei", "Einmalig", "Selten", "Häufig"]
    },
    color_discrete_map={
        "Frei": "#87CEFA",
        "Einmalig": "#FFD700",
        "Selten": "#FF6A6A",
        "Häufig": "#8B4513"
    }
)

fig_auf_genre.update_yaxes(tickformat=".0%")
fig_auf_genre.update_xaxes(tickangle=45)
fig_auf_genre.update_layout(
    legend_title_text="Auftreten",
    font=dict(size=22)
)

#fig_auf_genre.show()
