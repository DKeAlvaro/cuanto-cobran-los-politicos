import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import numpy as np

st.set_page_config(page_title='Salarios Públicos España', layout='wide')

@st.cache_data
def load_data():
    df = pd.read_csv('cleaned.csv', index_col=0)
    return df

df_full = load_data()
n_0_salary = len(df_full[df_full['currentAnnualSalary'] == 0])
df = df_full[df_full['currentAnnualSalary'] > 0]

st.title('Salarios de políticos en España') 


st.markdown(f'Excluyendo {n_0_salary} cargos con salario anual igual a 0, datos extraídos de [transparentia.newtral.es](https://transparentia.newtral.es/)', unsafe_allow_html=True)    
st.download_button(
    label='Descargar CSV',
    data=df.to_csv().encode('utf-8'),
    file_name='salarios_politicos.csv',
    mime='text/csv',
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Total de Cargos', f'{len(df):,}')
with col2:
    st.metric('Salario Medio Anual', f"€{df['currentAnnualSalary'].mean():,.0f}")
with col3:
    st.metric('Salario Máximo', f"€{df['currentAnnualSalary'].max():,.0f}")
with col4:
    st.metric('Suma Total de Salarios', f"€{df['currentAnnualSalary'].sum():,.0f}")



st.divider()

# --- Distribución de Salarios ---
st.subheader('Distribución de Salarios')

fig_histogram = go.Figure()

fig_histogram.add_trace(go.Histogram(
    x=df['currentAnnualSalary'],
    marker=dict(
        color='#6366f1',
        line=dict(width=1, color='black')
    ),
    xbins=dict(  
        start=0,
        end=150000,
        size=(150000 - 0) / 20
    ),
    hovertemplate=(
        'Salario: €%{x:,.0f}<br>' +
        'Frecuencia: %{y}<br>' +
        '<extra></extra>'
    )
))

median_salary = df['currentAnnualSalary'].median()
fig_histogram.add_vline(
    x=median_salary,
    line_dash='dash',
    line_color='red',
    annotation_text=f'Mediana: €{median_salary:,.0f}',
    annotation_position='top left'
)

mean_salary = df['currentAnnualSalary'].mean()
fig_histogram.add_vline(
    x=mean_salary,
    line_dash='dot',
    line_color='green',
    annotation_text=f'Media: €{mean_salary:,.0f}',
    annotation_position='top right'
)

fig_histogram.update_layout(
    xaxis_title='Salario Anual (€)',
    yaxis_title='Frecuencia',
    showlegend=False,
    height=400,
    xaxis=dict(range=[0, 150000])
)

st.plotly_chart(fig_histogram, width='stretch')
st.divider()



# --- Salario por Géneros ---
st.subheader('Salario por Géneros')

gender_labels = {'male': 'Hombres', 'female': 'Mujeres'}
df_plot = df.copy()

fig_combined = go.Figure()

colors = {'male': '#3b82f6', 'female': '#ec4899'}

for gender in ['male', 'female']:
    gender_data = df_plot[df_plot['gender'] == gender]
    label = gender_labels[gender]
    count = len(gender_data)
    
    kde = stats.gaussian_kde(gender_data['currentAnnualSalary'].dropna())
    x_range = np.linspace(0, 200000, 500)
    y_kde = kde(x_range)*10000
    
    hex_color = colors[gender].lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    rgba_fill = f'rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.1)'  
    
    fig_combined.add_trace(go.Scatter(
        x=x_range,
        y=y_kde,
        name=f'{label} (n={count:,})',
        line=dict(color=colors[gender], width=3),
        fill='tozeroy',
        fillcolor=rgba_fill,  
        opacity=1, 
        hovertemplate=(
            '<b>%{fullData.name}</b><br>' +
            'Salario: €%{x:,.2f}<br>' +
            'Densidad: %{y:.2f}<br>' +
            '<extra></extra>'
        )
    ))

for i, gender in enumerate(['male', 'female']):
    gender_data = df_plot[df_plot['gender'] == gender]
    label = gender_labels[gender]
    mean_salary = gender_data['currentAnnualSalary'].mean()
    
    if gender == 'male':
        annotation_pos = 'top'
        annotation_y = 1.02
    else:
        annotation_pos = 'bottom'
        annotation_y = 1.01
    
    fig_combined.add_vline(
        x=mean_salary,
        line_dash="dash",
        line_color=colors[gender],
        line_width=2,
        annotation_text=f'Media {label}: €{mean_salary:,.0f}',
        annotation_position=annotation_pos,
        annotation_font_color=colors[gender],
        annotation_y=annotation_y,
        opacity=0.8
    )

fig_combined.update_layout(
    xaxis_title='<b>Salario Anual (€)</b>',
    yaxis_title='<b>Densidad de Probabilidad</b>',
    showlegend=True,
    xaxis=dict(
        range=[0, 170000],
        tickformat='€,.0f'
    ),
    yaxis=dict(
        tickformat='.2f'
    ),
    font=dict(size=12),
    margin=dict(t=80, b=80),
    legend=dict(orientation='h', yanchor='bottom', y=-0.4, xanchor='center', x=0.5)
)

st.plotly_chart(fig_combined, width='stretch')
st.divider()

# --- Salario Medio por Nivel Administrativo ---
st.subheader('Salario Medio por Nivel Administrativo')

salary_by_admin_level = (
    df.groupby('spatial_name')['currentAnnualSalary']
      .agg(['mean', 'count'])
      .sort_values('mean', ascending=False)
)
percent_by_admin_level = (salary_by_admin_level['count'] / salary_by_admin_level['count'].sum()) * 100

fig = make_subplots(rows=1, cols=1, specs=[[{'secondary_y': True}]])

fig.add_trace(
    go.Bar(
        x=salary_by_admin_level.index,
        y=salary_by_admin_level['mean'],
        name='Salario medio',
        text=salary_by_admin_level['mean'],
        texttemplate='€%{text:,.0f}',
        textposition='outside',
        marker=dict(color=salary_by_admin_level['mean'], colorscale='Blues', showscale=False),
        hovertemplate=(
            '<b>%{x}</b><br>' +
            'Salario medio: €%{y:,.0f}<br>' +
            'Empleados: %{customdata}<br>' +
            '<extra></extra>'
        ),
        customdata=salary_by_admin_level['count']
    ),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=salary_by_admin_level.index,
        y=percent_by_admin_level,
        name='Porcentaje de empleados',
        mode='lines+markers',
        marker=dict(color='rgba(200, 0, 0, 0.8)', size=8),
        line=dict(color='rgba(200, 0, 0, 0.6)', width=2),
        hovertemplate=(
            '<b>%{x}</b><br>' +
            'Porcentaje: %{y:.1f}%<br>' +
            'Empleados: %{customdata}<br>' +
            '<extra></extra>'
        ),
        customdata=salary_by_admin_level['count']
    ),
    secondary_y=True
)

fig.update_xaxes(title_text='Nivel administrativo', showgrid=False)
fig.update_yaxes(title_text='Salario Medio Anual (€)', secondary_y=False, showgrid=False)
fig.update_yaxes(title_text='Porcentaje de Empleados (%)', secondary_y=True, showgrid=False)
fig.update_layout(height=500, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1.0))

st.plotly_chart(fig, width='stretch')
st.divider()

# --- Salario Medio por Tipo de Cargo ---
st.subheader('Salario Medio por Tipo de Cargo')
st.markdown('Cada punto es un cargo diferente, prueba a tocar uno')
salary_by_role = (
    df.groupby('role_name')['currentAnnualSalary']
      .agg(mean='mean', count='count')
      .sort_values('mean', ascending=False)
      .reset_index()
)

fig_role_scatter = px.scatter(
    salary_by_role,
    x='mean',
    y='count',
    hover_name='role_name',
    color='mean',
    color_continuous_scale='Viridis',
    labels={'mean': 'Salario Medio Anual (€)', 'count': 'Número de Empleados'},
)

fig_role_scatter.update_traces(
    hovertemplate=(
        '<b>%{hovertext}</b><br>' +
        'Salario medio: €%{x:,.0f}<br>' +
        'Empleados: %{y}<br>' +
        '<extra></extra>'
    )
)
fig_role_scatter.update_traces(marker=dict(size=10, opacity=0.85))
fig_role_scatter.update_layout(
    height=500,
    showlegend=False,
    xaxis=dict(tickformat='€,.0f'),
    coloraxis_showscale=False
)

st.plotly_chart(fig_role_scatter, width='stretch')

st.divider()

affiliation_names = {
    'partido-popular': 'Partido Popular',
    'psoe': 'PSOE',
    'sin-partido': 'Sin Partido',
    'cm': 'Coalición Madrid',
    'vox': 'VOX',
    'eh-bildu': 'EH Bildu',
    'erc': 'ERC',
    'eaj-pnv': 'EAJ-PNV',
    'junts-per-catalunya': 'Junts',
    'par': 'PAR',
    'podemos': 'Podemos'
}

# --- Salario Medio por Partido Político ---
st.subheader('Partidos Políticos con mayor salario medio')
st.write('Mostrando partidos con 5 o más integrantes')
salary_by_affiliation = df.groupby('affiliation_slug').agg(
    mean_salary=('currentAnnualSalary', 'mean'),
    count=('currentAnnualSalary', 'count')
).query('count >= 5').sort_values('mean_salary', ascending=False)

salary_by_affiliation['party_name'] = salary_by_affiliation.index.map(lambda x: affiliation_names.get(x, x.upper()))

df_display = salary_by_affiliation[['party_name', 'mean_salary']].reset_index(drop=True).rename(columns={
    'party_name': 'Partido Político',
    'mean_salary': 'Salario Medio'
})

df_display['Salario Medio'] = df_display['Salario Medio'].apply(lambda x: f"€{x:,.0f}")

st.dataframe(
    df_display,
    hide_index=True
)
st.divider()

# --- Mapa Interactivo de España ---
st.subheader('Mapa Interactivo de España')

st.markdown('Explora el salario medio de alcaldes por municipio')

geo_df = pd.read_csv('georef-spain-municipio.csv', delimiter=';')[['Official Name Municipality', 'Geo Point']]
geo_df = geo_df.rename(columns={'Official Name Municipality': 'municipality'})

mayors_df = df_full[df_full['role_name'] == 'Alcalde'].copy()
mayors_df['municipality'] = mayors_df['member_of_name'].str.replace(r'Ayuntamiento De\s*', '', regex=True)
mayors_df['municipality'] = mayors_df['municipality'].apply(lambda x: x.split('/')[0] if '/' in str(x) else x)

exclude_zero_salary = st.checkbox('Excluir salarios de 0€', value=True)
if exclude_zero_salary:
    mayors_df = mayors_df[mayors_df['currentAnnualSalary'] > 0]

mayor_locations = mayors_df['municipality'].tolist()
map_municipalities = geo_df['municipality'].tolist()
missing_municipalities = [m for m in mayor_locations if m not in map_municipalities]
present_count = len(mayor_locations) - len(missing_municipalities)

municipality_agg = mayors_df.groupby('municipality').agg(
    mean_salary=('currentAnnualSalary', 'mean'),
    count=('currentAnnualSalary', 'count'),
    max_salary=('currentAnnualSalary', 'max')
).reset_index()

map_df = pd.merge(municipality_agg, geo_df, on='municipality', how='inner')
coordinates = map_df['Geo Point'].str.split(',', expand=True)
map_df['lat'] = pd.to_numeric(coordinates[0], errors='coerce')
map_df['lon'] = pd.to_numeric(coordinates[1], errors='coerce')
map_df = map_df.dropna(subset=['lat', 'lon'])

mayor_info = (
    mayors_df[['municipality', 'name', 'currentAnnualSalary']]
    .drop_duplicates(subset=['municipality'])
)
mayor_info = mayor_info.rename(columns={'name': 'mayor_name', 'currentAnnualSalary': 'mayor_salary'})
map_df = pd.merge(map_df, mayor_info, on='municipality', how='left')

fig_map = px.scatter_map(
    map_df,
    lat='lat',
    lon='lon',
    color='mean_salary',
    size='count',
    hover_name='municipality',
    hover_data={
        'mayor_name': True,
        'mayor_salary': ':,.0f',
        'mean_salary': False,
        'count': False,
        'max_salary': False,
        'lat': False,
        'lon': False,
        'municipality': False
    },
    color_continuous_scale='Viridis', 
    size_max=6,
    zoom=5.2,
    height=700,
    labels={
        'mean_salary': 'Salario Medio (€)',
        'count': 'Número de Cargos',
        'mayor_salary': 'Salario (€)',
        'mayor_name': 'Alcalde'

    }
)

fig_map.update_layout(
    mapbox_style='carto-positron',
    margin=dict(l=0, r=0, t=0, b=0),
    coloraxis_colorbar=dict(
        title='Salario (€)',
        orientation='h',
        yanchor='bottom',
        y=-0.1,
        xanchor='center',
        x=0.5
    )
)

st.plotly_chart(fig_map, width='stretch')

st.metric('Municipios con datos', present_count)


st.divider()

# --- Explorar Datos ---
st.subheader('Explorar Datos')
search_term = st.text_input('Buscar en cualquier columna:')
filtered_df = df_full.copy()
if search_term:
    term = search_term.lower()
    mask = pd.Series(False, index=filtered_df.index)
    for col in filtered_df.columns:
        col_series = filtered_df[col].astype(str).str.lower().str.contains(term, regex=False)
        mask = mask | col_series
    filtered_df = filtered_df[mask]

st.write(f'Resultados encontrados: {len(filtered_df)}')
if len(filtered_df) > 0:
    st.dataframe(
        filtered_df[['name', 'currentAnnualSalary', 'jobTitle', 'role_name', 'affiliation_slug', 'member_of_name']].sort_values('currentAnnualSalary', ascending=False),
        width='stretch',
        hide_index=True
    )