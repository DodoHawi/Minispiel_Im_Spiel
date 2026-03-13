import pandas as pd
import plotly.express as px


# --- Komplexität-Plot ---
# Neue CSV einlesen
df_komp = pd.read_csv('codebook_database - database.csv', sep=',')

# Whitespace in Spaltennamen entfernen
df_komp.columns = df_komp.columns.str.strip()

# Absolute Anzahl jeder Komplexität insgesamt
komp_counts = df_komp['Cooldown'].value_counts().reindex(['ja', 'nein', 'n']).reset_index()
komp_counts.columns = ['Cooldown', 'Anzahl']

# Plot: Verteilung der Komplexität (Pie Chart)
fig_komp_count = px.pie(
    komp_counts,
    values='Anzahl',
    names='Cooldown',
    title='Verteilung der Komplexität-Level',
    color='Cooldown',
    color_discrete_map={'ja': '#87CEFA', 'nein': '#FFD700', 'n': "#0CF948"}
)

fig_komp_count.update_traces(textinfo='percent+value')

fig_komp_count.update_layout(
    font=dict(size=25), 
    legend=dict(
        title=dict(text = "Cooldown")))



fig_komp_count.show()
