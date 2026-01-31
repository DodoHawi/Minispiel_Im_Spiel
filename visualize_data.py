from matplotlib.pyplot import title
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots   

df = pd.read_csv('minispiele_raw_clean.csv', sep=';')

# Spalten mit nur Ja/Nein/NaN
binary_columns = [
    col for col in df.columns
    if df[col].dropna().isin(['Ja', 'Nein']).all()
]

# Ja/Nein in 1/0 umwandeln
df_bin = df.copy()
for col in binary_columns:
    df_bin[col] = df_bin[col].map({'Ja': 1, 'Nein': 0}) 
    
# Wide to long format
df_long = df_bin.melt(
    id_vars=["Spiel", "Genre", "Minispiel"],
    value_vars=binary_columns,
    var_name="feature",
    value_name="present"
)

# Anteil "Ja" pro Feature berechnen
feature_summary = (
    df_long.groupby("feature")["present"]
    .mean()
    .reset_index()
.rename(columns={"present": "Anteil_Ja", "feature": "Kategorie"})
)

# Ergänzen des Anteils "Nein"
feature_summary["Anteil_Nein"] = 1 - feature_summary["Anteil_Ja"]

# Reshape für Plotting
feature_summary_melted = feature_summary.melt(
    id_vars=["Kategorie"],
    value_vars=["Anteil_Ja", "Anteil_Nein"],
    var_name="Ergebnis",
    value_name="Anteil"
)

# Vergleich nach Genre
genre_share = (
    df_long.groupby(["Genre", "feature"])["present"]
    .mean()
    .reset_index(name="Anteil")
)

# Plot der Ergebnisse
fig1 = px.bar(
    feature_summary_melted,
    x="Anteil",
    y="Kategorie",
    color="Ergebnis",
    orientation='h',
    title="Ergebnis pro Kategorie",
    labels={"Anteil": "Anteil", "Kategorie": "Kategorie", "Ergebnis": "Ergebnis"},
    color_discrete_map={"Anteil_Ja": "#87CEFA", "Anteil_Nein": "#FF6A6A"}
)

fig2 = px.bar(
    genre_share,
    x="Anteil",
    y="feature",
    color="Genre",
    orientation='h',
    barmode='group',
    title="Anteil 'Ja' pro Kategorie und Genre",
)

fig3 = px.bar(
    genre_share,
    x="Genre",
    y="Anteil", 
    facet_col="feature",
    facet_col_wrap=3,
    labels={"Anteil": "Anteil 'Ja'", "Genre": "Genre", "feature": "Kategorie"},
    title="Anteil 'Ja' pro Kategorie und Genre (Facettenansicht)"
)

#fig1.show()
#fig2.show()
#fig3.show()

# --- Komplexität-Plot ---
# Neue CSV einlesen
df_komp = pd.read_csv('minispiele_komplexitaet_clean.csv', sep=';')

# Whitespace in Spaltennamen entfernen
df_komp.columns = df_komp.columns.str.strip()

# Absolute Anzahl jeder Komplexität insgesamt
komp_counts = df_komp['Komplexität'].value_counts().reindex(['Hoch', 'Mittel', 'Gering']).reset_index()
komp_counts.columns = ['Komplexität', 'Anzahl']

# Plot: Verteilung der Komplexität (Pie Chart)
fig_komp_count = px.pie(
    komp_counts,
    values='Anzahl',
    names='Komplexität',
    title='Verteilung der Komplexität-Level (Anteile)',
    color='Komplexität',
    color_discrete_map={'Gering': '#87CEFA', 'Mittel': '#FFD700', 'Hoch': '#FF6A6A'}
)
#fig_komp_count.show()

# ---------------------------------------------------------------------------------------------------
# --- Auftreten-Plot ---
# ---------------------------------------------------------------------------------------------------

# Neue CSV einlesen
df_auf = pd.read_csv('minispiele_auftreten_clean.csv', sep=';')

# Whitespace in Spaltennamen entfernen
df_auf.columns = df_auf.columns.str.strip()

# Absolute Anzahl jeder Komplexität insgesamt
auf_counts = df_auf['Auftreten'].value_counts().reindex(['Frei','Einmalig', 'Selten', 'Häufig']).reset_index()
auf_counts.columns = ['Auftreten', 'Anzahl']

# Plot: Verteilung des Auftretens (Pie Chart)
fig_auf_count = px.pie(
    auf_counts,
    values='Anzahl',
    names='Auftreten',
    title='Verteilung der Auftreten-Level (Anteile)',
    color='Auftreten',
    color_discrete_map={'Frei': '#87CEFA', 'Einmalig': '#FFD700', 'Selten': '#FF6A6A', 'Häufig': '#8B4513'}
)

# fig_auf_count.show()

# ---------------------------------------------------------------------------------------------------
# --- UI-Transformation-Plot ---
# ---------------------------------------------------------------------------------------------------

# Neue CSV einlesen
df_ui = pd.read_csv('minispiele_uitransformation_clean.csv', sep=';')

# Whitespace in Spaltennamen entfernen
df_ui.columns = df_ui.columns.str.strip()

# Absolute Anzahl jeder UI-Transformation insgesamt
ui_counts = df_ui['UI-Transformation'].value_counts().reindex(['Keine','Leicht', 'Mittel', 'Stark']).reset_index()
ui_counts.columns = ['UI-Transformation', 'Anzahl']

# Plot: Verteilung der UI-Transformation (Pie Chart)
fig_ui_count = px.pie(
    ui_counts,
    values='Anzahl',
    names='UI-Transformation',
    title='Verteilung der UI-Transformation (Anteile)',
    color='UI-Transformation',
    color_discrete_map={'Keine': '#87CEFA', 'Leicht': '#FFD700', 'Mittel': '#FF6A6A', 'Stark': '#8B4513'}
)

# fig_ui_count.show()

# ---------------------------------------------------------------------------------------------------
# --- Veränderung der Schwierigkeit-Plot ---
# ---------------------------------------------------------------------------------------------------

# Neue CSV einlesen
df_ver = pd.read_csv('minispiele_veraenderung_clean.csv', sep=';')

# Whitespace in Spaltennamen entfernen
df_ver.columns = df_ver.columns.str.strip()

# Absolute Anzahl jeder Veränderung der Schwierigkeit insgesamt
ver_counts = df_ver['Veränderung der Schwierigkeit'].value_counts().reindex(['Konstant','Kontext: Leichter', 
    'Kontext: Schwieriger', 'Minispiel: Leichter', 'Minispiel: Schwieriger']).reset_index()
ver_counts.columns = ['Veränderung der Schwierigkeit', 'Anzahl']

# Rename legend values
ver_counts['Veränderung der Schwierigkeit'] = ver_counts['Veränderung der Schwierigkeit'].replace({
    'Kontext: Leichter': 'K: Leichter',
    'Kontext: Schwieriger': 'K: Schwieriger',
    'Minispiel: Leichter': 'M: Leichter',
    'Minispiel: Schwieriger': 'M: Schwieriger'
})

# Plot: Verteilung der Veränderung der Schwierigkeit (Pie Chart)
fig_ver_count = px.pie(
    ver_counts,
    values='Anzahl',
    names='Veränderung der Schwierigkeit',
    title='Verteilung der Veränderung der Schwierigkeit (Anteile)',
    color='Veränderung der Schwierigkeit',
    color_discrete_map={'Konstant': '#87CEFA', 'K: Leichter': '#FFD700', 'K: Schwieriger': '#FF6A6A', 'M: Leichter': '#8B4513', 'M: Schwieriger': '#000000'}
)

# fig_ver_count.show()

# ---------------------------------------------------------------------------------------------------
# --- Anzeigen aller Subplots in einem Plot ---
# ---------------------------------------------------------------------------------------------------
fig_grid = make_subplots(
    rows=2, cols=2,
    specs=[[{"type":"domain"}, {"type":"domain"}],
           [{"type":"domain"}, {"type":"domain"}]],
    subplot_titles=(
        'Komplexität',
        'Auftreten',
        'UI-Transformation',
        'Veränderung der Schwierigkeit'
    ),
    shared_xaxes=False,
    shared_yaxes=False
)
# Hinzufügen der einzelnen Subplots
for trace in fig_komp_count.data:
    fig_grid.add_trace(trace, row=1, col=1)
for trace in fig_auf_count.data:
    fig_grid.add_trace(trace, row=1, col=2)
for trace in fig_ui_count.data:
    fig_grid.add_trace(trace, row=2, col=1)
for trace in fig_ver_count.data:
    fig_grid.add_trace(trace, row=2, col=2)

# Update layout as needed
fig_grid.update_layout(height=900, width=1250, showlegend=False, title_text="Vergleichende Übersicht")

# Helper to draw a small legend (colored boxes + labels) below each subplot in paper coordinates
def add_sub_legend(fig, x_center, y_base, items, box_w=0.02, box_h=0.02, spacing=0.1):
    x = x_center - (len(items)-1)*spacing/2
    for label, color in items:
        fig.add_shape(type="rect",
                      xref="paper", yref="paper",
                      x0=x-box_w/2, x1=x+box_w/2,
                      y0=y_base, y1=y_base+box_h,
                      fillcolor=color, line=dict(color=color))
        fig.add_annotation(x=x+box_w/2+0.01, y=y_base+box_h/2,
                           xref="paper", yref="paper",
                           text=label, showarrow=False, xanchor="left", yanchor="middle", font=dict(size=12))
        x += spacing

# Define legend items (label, color)
comp_items = [('Hoch', '#FF6A6A'), ('Mittel', '#FFD700'), ('Gering', '#87CEFA')]
auf_items = [('Frei','#87CEFA'), ('Einmalig','#FFD700'), ('Selten','#FF6A6A'), ('Häufig','#8B4513')]
ui_items = [('Keine','#87CEFA'), ('Leicht','#FFD700'), ('Mittel','#FF6A6A'), ('Stark','#8B4513')]
ver_items = [('Konstant','#87CEFA'), ('Kontext: -','#FFD700'), ('Kontext: +','#FF6A6A'), ('Minisp.: -','#8B4513'), ('Minisp.: +','#000000')]

# Add legends under each subplot (paper coords)
add_sub_legend(fig_grid, x_center=0.18, y_base=0.55, items=comp_items)
add_sub_legend(fig_grid, x_center=0.7, y_base=0.55, items=auf_items)
add_sub_legend(fig_grid, x_center=0.18, y_base=0.45, items=ui_items)
add_sub_legend(fig_grid, x_center=0.7, y_base=0.45, items=ver_items)

fig_grid.show()
