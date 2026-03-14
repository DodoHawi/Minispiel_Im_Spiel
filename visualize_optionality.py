import pandas as pd
import plotly.express as px


# ---------------------------------------------------------------------------------------------------
# --- Optionalität ---
# ---------------------------------------------------------------------------------------------------
df_opt = pd.read_csv('codebook_database - database.csv', sep=',')

# Whitespace in Spaltennamen entfernen
df_opt.columns = df_opt.columns.str.strip()

# Absolute Anzahl Optionalität insgesamt
opt_counts = df_opt['Optionalität'].value_counts().reindex(['ja','nein', 'einmalig']).reset_index()
opt_counts.columns = ['Optionalität', 'Anzahl']

# Rename legend values
opt_counts['Optionalität'] = opt_counts['Optionalität'].replace({
    'ja': 'Ja',
    'nein': 'Nein',
    'einmalig': 'Einmalig'
})

# Plot: Verteilung der Optionalität (Pie Chart)
fig_opt_count = px.pie(
    opt_counts,
    values='Anzahl',
    names='Optionalität',
    title='Verteilung der Optionalität',
    color='Optionalität',
    color_discrete_map={'Ja': '#87CEFA', 'Einmalig': '#FFD700', 'Nein': '#FF6A6A'}
)

fig_opt_count.update_traces(textinfo='percent+value')

fig_opt_count.update_layout(
    font=dict(size=25), 
    legend=dict(
        title=dict(text = "Optionalität")))

fig_opt_count.show()