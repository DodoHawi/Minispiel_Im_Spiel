import pandas as pd
import plotly.express as px

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
ver_counts.columns = ['Veränderung der Schwierigkeit', 'Anzahl']

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
    values='Anzahl',
    names='Veränderung der Schwierigkeit',
    title='Verteilung der Veränderung der Schwierigkeit',
    color='Veränderung der Schwierigkeit',
    color_discrete_map={'Konstant': '#87CEFA', 'Kontext: Leichter': '#FFD700', 
    'Kontext: Schwieriger': '#FF6A6A', 'Minispiel: Leichter': "#69138B", 
    'Minispiel: Schwieriger': "#8EFF0E",
    'Kontext: Situativ': '#FF00F7'}
)

fig_ver_count.update_traces(textinfo='percent+value')

fig_ver_count.update_layout(
    font=dict(size=25), 
    legend=dict(
        title=dict(text = "Veränderung der Schwierigkeit")))

fig_ver_count.show()