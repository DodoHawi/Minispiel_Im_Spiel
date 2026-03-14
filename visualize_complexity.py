import pandas as pd
import plotly.express as px

# --- Komplexität-Plot ---
# Neue CSV einlesen
df_komp = pd.read_csv('codebook_database - database.csv', sep=',')

# Whitespace in Spaltennamen entfernen
df_komp.columns = df_komp.columns.str.strip()

# Absolute Anzahl jeder Komplexität insgesamt
komp_counts = df_komp['Komplexität'].value_counts().reindex(['hoch', 'mittel', 'gering']).reset_index()
komp_counts['Komplexität'] = komp_counts['Komplexität'].replace(
    {'hoch': 'Hoch', 'mittel': 'Mittel', 'gering': 'Gering'})
komp_counts.columns = ['Komplexität', 'Anzahl']

# Plot: Verteilung der Komplexität (Pie Chart)
fig_komp_count = px.pie(
    komp_counts,
    values='Anzahl',
    names='Komplexität',
    title='Verteilung der Komplexität-Level',
    color='Komplexität',
    color_discrete_map={'Gering': '#87CEFA', 'Mittel': '#FFD700', 'Hoch': '#FF6A6A'}
)

fig_komp_count.update_traces(textinfo='percent+value')

fig_komp_count.update_layout(
    font=dict(size=25), 
    legend=dict(
        title=dict(text = "Komplexität")))



fig_komp_count.show()