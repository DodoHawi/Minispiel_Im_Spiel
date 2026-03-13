from matplotlib.pyplot import title
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots   
from sklearn.preprocessing import MultiLabelBinarizer  


# ---------------------------------------------------------------------------------------------------
# --- UI-Transformation-Plot ---
# ---------------------------------------------------------------------------------------------------

# Neue CSV einlesen
df_ui = pd.read_csv('codebook_database - database.csv', sep=',')

# Whitespace in Spaltennamen entfernen
df_ui.columns = df_ui.columns.str.strip()

# Absolute Anzahl jeder UI-Transformation insgesamt
ui_counts = df_ui['UI - Transformation'].value_counts().reindex(['keine','leicht', 'mittel', 'stark']).reset_index()
ui_counts['UI - Transformation'] = ui_counts['UI - Transformation'].replace({
    'keine': 'Keine',
    'leicht': 'Leicht',
    'mittel': 'Mittel',
    'stark': 'Stark'
})
ui_counts.columns = ['UI - Transformation', 'Anzahl']

# Plot: Verteilung der UI-Transformation (Pie Chart)
fig_ui_count = px.pie(
    ui_counts,
    values='Anzahl',
    names='UI - Transformation',
    title='Verteilung der UI-Transformation',
    color='UI - Transformation',
    color_discrete_map={'Keine': '#87CEFA', 'Leicht': '#FFD700', 'Mittel': '#FF6A6A', 'Stark': '#8B4513'}
)

fig_ui_count.update_traces(textinfo='percent+value')

fig_ui_count.update_layout(
    font=dict(size=25), 
    legend=dict(
        title=dict(text = "UI-Transformationsgrad")))

fig_ui_count.show()

#----------------------------------------------------------------------------------------------------
# UI-Transformation nach Genre: Gruppieren und Plotten
# ----------------------------------------------------------------------------------------------------

# Anzahl pro Genre × Komplexität
ui_by_genre = (
    df_ui
    .groupby(["Genre", "UI-Transformation"])
    .size()
    .reset_index(name="count")
)

# Anteile innerhalb jedes Genres
ui_by_genre["Anteil"] = (
    ui_by_genre["count"]
    / ui_by_genre.groupby("Genre")["count"].transform("sum")
)


fig_ui_genre = px.bar(
    ui_by_genre,
    x="Genre",
    y="Anteil",
    color="UI-Transformation",
    title="UI-Transformation der Minispiele nach Genre",
    category_orders={
        "UI-Transformation": ["Keine", "Leicht", "Mittel", "Stark"]
    },
    color_discrete_map={
        "Keine": "#87CEFA",
        "Leicht": "#FFD700",
        "Mittel": "#FF6A6A",
        "Stark": "#8B4513"
    }
)

fig_ui_genre.update_yaxes(tickformat=".0%")
fig_ui_genre.update_xaxes(tickangle=45)
fig_ui_genre.update_layout(
    legend_title_text="UI-Transformation",
    font=dict(size=22)
)

#fig_ui_genre.show()