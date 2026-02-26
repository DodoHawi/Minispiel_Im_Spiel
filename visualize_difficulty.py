import ast
from matplotlib.pyplot import title
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots   
from sklearn.preprocessing import MultiLabelBinarizer  

# ---------------------------------------------------------------------------------------------------
# --- Veränderung der Schwierigkeit-Plot ---
# ---------------------------------------------------------------------------------------------------

# Neue CSV einlesen
df_ver = pd.read_csv('codebook_database - database.csv', sep=',')

# Whitespace in Spaltennamen entfernen
df_ver.columns = df_ver.columns.str.strip()

# Absolute Anzahl jeder Veränderung der Schwierigkeit insgesamt
ver_counts = df_ver['Veränderung der Schwierigkeit'].value_counts().reindex(['konstant', 
    'minispiel_leichter', 'minispiel_schwieriger', 'kontext_leichter', 'kontext_schwieriger', 
    'kontext_situativ']).reset_index()
ver_counts.columns = ['Veränderung der Schwierigkeit', 'Anteil']

# Rename legend values
ver_counts['Veränderung der Schwierigkeit'] = ver_counts['Veränderung der Schwierigkeit'].replace({
    'kontext_leichter': 'Kontext: Leichter',
    'kontext_schwieriger': 'Kontext: Schwieriger',
    'minispiel_leichter': 'Minispiel: Leichter',
    'minispiel_schwieriger': 'Minispiel: Schwieriger',
    'kontext_situativ': 'Kontext: Situativ',
    'konstant': 'Konstant'
})

# Plot: Verteilung der Veränderung der Schwierigkeit (Pie Chart)
fig_ver_count = px.pie(
    ver_counts,
    values='Anteil',
    names='Veränderung der Schwierigkeit',
    title='Verteilung der Veränderung der Schwierigkeit',
    color='Veränderung der Schwierigkeit',
    color_discrete_map={'Konstant': '#87CEFA', 'Kontext: Leichter': '#FFD700', 
    'Kontext: Schwieriger': '#FF6A6A', 'Minispiel: Leichter': "#69138B", 
    'Minispiel: Schwieriger': "#8EFF0E",
    'Kontext: Situativ': '#FF00F7'}
)

fig_ver_count.update_layout(
    font=dict(size=25), 
    legend=dict(
        title=dict(text = "Veränderung der Schwierigkeit")))

fig_ver_count.show()

#----------------------------------------------------------------------------------------------------
# Veränderung der Schwierigkeit nach Genre: Gruppieren und Plotten
# ----------------------------------------------------------------------------------------------------

# Anzahl pro Genre × Komplexität
ver_by_genre = (
    df_ver
    .groupby(["Genre", "Veränderung der Schwierigkeit"])
    .size()
    .reset_index(name="count")
)

# Anteile innerhalb jedes Genres
ver_by_genre["Anteil"] = (
    ver_by_genre["count"]
    / ver_by_genre.groupby("Genre")["count"].transform("sum")
)


fig_ver_genre = px.bar(
    ver_by_genre,
    x="Genre",
    y="Anteil",
    color="Veränderung der Schwierigkeit",
    title="Veränderung der Schwierigkeit der Minispiele nach Genre",
    category_orders={
        "Veränderung der Schwierigkeit": ["Konstant", "Kontext: Leichter", "Kontext: Schwieriger", "Minispiel: Leichter", "Minispiel: Schwieriger"]
    },
    color_discrete_map={
        "Konstant": "#87CEFA",
        "Kontext: Leichter": "#FF00F7",
        "Kontext: Schwieriger": "#FF6A6A",
        "Minispiel: Leichter": "#8B4513",
        "Minispiel: Schwieriger": "#420EFF"
    }
)

fig_ver_genre.update_yaxes(tickformat=".0%")
fig_ver_genre.update_xaxes(tickangle=45)
fig_ver_genre.update_layout(
    legend_title_text="Veränderung der Schwierigkeit",
    font=dict(size=22)
)

#fig_ver_genre.show()
