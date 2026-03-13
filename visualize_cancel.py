import pandas as pd
import plotly.express as px


# ---------------------------------------------------------------------------------------------------
# --- Abbruchbarkeit ---
# ---------------------------------------------------------------------------------------------------
df_opt = pd.read_csv('codebook_database - database.csv', sep=',')

# Whitespace in Spaltennamen entfernen
df_opt.columns = df_opt.columns.str.strip()

# Absolute Anzahl Abbruchbarkeit insgesamt
opt_counts = df_opt['Abbruchbarkeit'].value_counts().reindex(['ja','nein', 'ja_bereich']).reset_index()
opt_counts.columns = ['Abbruchbarkeit', 'Anzahl']

# Rename legend values
opt_counts['Abbruchbarkeit'] = opt_counts['Abbruchbarkeit'].replace({
    'ja': 'Ja',
    'nein': 'Nein',
    'ja_bereich': 'Ja_Bereich'
})

# Plot: Verteilung der Abbruchbarkeit (Pie Chart)
fig_opt_count = px.pie(
    opt_counts,
    values='Anzahl',
    names='Abbruchbarkeit',
    title='Verteilung der Abbruchbarkeit',
    color='Abbruchbarkeit',
    color_discrete_map={'Ja': '#87CEFA', 'Ja_Bereich': '#FFD700', 'Nein': '#FF6A6A'}
)

fig_opt_count.update_traces(textinfo='percent+value')

fig_opt_count.update_layout(
    font=dict(size=25), 
    legend=dict(
        title=dict(text = "Abbruchbarkeit")))

fig_opt_count.show()
