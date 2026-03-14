import pandas as pd
import plotly.express as px

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
    'selten': 'Selten (<5)',
    'häufig': 'Häufig (≥5)'
})
auf_counts.columns = ['Auftreten', 'Anzahl']

# Plot: Verteilung des Auftretens (Pie Chart)
fig_auf_count = px.pie(
    auf_counts,
    values='Anzahl',
    names='Auftreten',
    title='Verteilung der Auftreten-Level',
    color='Auftreten',
    color_discrete_map={'Frei': '#87CEFA', 'Einmalig': '#FFD700', 'Selten (<5)': '#FF6A6A', 'Häufig (≥5)': '#8B4513'}
)

fig_auf_count.update_traces(textinfo='percent+value')

fig_auf_count.update_layout(
    font=dict(size=25), 
    legend=dict(
        title=dict(text = "Auftreten")))

fig_auf_count.show()