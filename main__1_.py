import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px  # Import plotly.express
import pandas as pd
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import numpy as np
from math import sqrt


app = dash.Dash(__name__, suppress_callback_exceptions=True, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],)
app.title = "US Chronic disease Analytics Dashboard"
server = app.server

url = "https://storage.googleapis.com/mbcc/datasets/us_chronic_disease_indicators.csv"
df = pd.read_csv(url)

visualization_df = df.copy()

# Define the layout of the app
app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
       dcc.Tab(label='Visualizations-1',value='tab-1',style={'color': '#006400', 'padding': '6px'}),
       dcc.Tab(label='Visualizations-2', value='tab-2',style={'color': '#006400', 'padding': '6px'}),
       dcc.Tab(label='Prediction', value='tab-3',style={'color': '#006400', 'padding': '6px'}),
       # You can add more tabs here
    ],style={'height': '60px','alignItems': 'center', }, # Adjust the height accordingly
        parent_style={'display': 'flex','flexDirection': 'row','justifyContent': 'start',  'alignItems': 'center','height': '44px'  }
             ),
    html.Div(id='tabs-content'),
    # Added a comma here to separate elements
   dcc.Graph(id='first-graph', style={'display': 'none'}), 
   dcc.Graph(id='second-graph', style={'display': 'none'}), 
    dcc.Graph(id='third-graph', style={'display': 'none'}), 
    dcc.Graph(id='fourth-graph', style={'display': 'none'}), 
   dcc.Graph(id='fifth-graph', style={'display': 'none'}), 
])

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])

def render_content(tab):
    if tab =='tab-3':
       return html.Div([
            
            html.H3('Predict Chronic Disease Indicators for 2023'),
            html.H5('Use the Location code chart below for reference:'),
            html.Pre(state_codes_str, style={'whiteSpace': 'pre-wrap', 'wordBreak': 'break-all'}),
            dcc.Input(id='input-state', type='text', placeholder='Enter State Abbreviation', style={'marginRight': '10px'}),
            html.Button('Predict', id='predict-button', n_clicks=0),
            html.Div(id='prediction-result')
        ])
            # You can add more elements as needed

    if tab == 'tab-1':
        # You can return the content for tab-1 here, such as a different set of dropdowns and graphs or other information
        return html.Div([
            
    #first dropdown and its graph
        html.Div([
        # Left column for description and dropdown
        html.Div(
            [
                html.H1('US Chronic Disease Analytics Dashboard', style={'textAlign': 'center', 'color': '#007BFF'}),
                html.P("Select from Dropdown-1"),
                dcc.Dropdown(
                    id='first-dropdown',
                    options=[
                     {'label': 'Missing data summary', 'value': 'MDS'},
                     {'label': 'Distribution of Topic IDs', 'value': 'DTI'},
                     {'label': 'Top 5 data sources and their ratios', 'value': 'DS'},
                     {'label': 'Top 7 datavaluecount by count', 'value': 'DVC'},
                     {'label': '3D Scatterplot ', 'value': 'SP'}
                    ],
                    value='MDS'  # default value
                ),
            ],
            style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        html.Div([
            dcc.Graph(id='first-graph')
        ], style={'width': '70%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'padding': '20px'}),

    # Second dropdown and its graph
       html.Div([
        html.Div([
            html.P("Select from Dropdown 2:"),
            dcc.Dropdown(
                id='second-dropdown',
                options=[
                    {'label': 'Top 5 Diseases in NY', 'value': 'NY'},
                    {'label': 'Total Cases of Top 4 diseases by Gender in FL', 'value': 'FL'},
                    {'label': 'Diabetes Cases in PA by Year and Gender (2015-2021)', 'value': 'PA'},
                    {'label': 'Total Cardiovascular Disease Cases by Gender in AR Over the Years', 'value': 'AR'},
                    {'label': 'Total Cases of top diseases by Stratification in AR', 'value': 'TAR'}
                ],
                value='NY'  # Default value
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='second-graph')
        ], style={'width': '70%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'flex-direction': 'row','padding':'20px'}),

 #third dropdown and its graph
    html.Div([
        html.Div([
            html.P("Select from Dropdown 3:"),
            dcc.Dropdown(
                id='third-dropdown',
                options=[
                        {'label': 'Total Cases of cancer Over the Years by Location', 'value': 'LOC'},
                        {'label': 'Total Cases of diseases Over Time in MI', 'value': 'MI'},
                        {'label': 'Total Cases per disease over time in NY', 'value': 'NY'},
                        {'label': 'Total cancer Cases in FL Over the Years', 'value': 'FL'},
                        {'label': 'Total Cases of Diseases Over Time in in AZ', 'value': 'AZ'}
                ],
                value='LOC'  # Default value
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='third-graph')
        ], style={'width': '70%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'flex-direction': 'row','padding':'20px'}),

])

    elif tab == 'tab-2':
        return html.Div([
            # Fourth dropdown and its graph
        html.Div([
        # Left column for description and dropdown
        html.Div(
            [

                html.P("Select from Dropdown 4:"),
                dcc.Dropdown(
                    id='fourth-dropdown',
                    options=[
                        {'label': 'Total Cases by State ', 'value': 'TC'},
                        {'label': 'Total Arthritis Cases by State', 'value': 'AR'},
                        {'label': 'Total Cancer Cases by State', 'value': 'CA'},
                        {'label': 'Total Alcohol Cases by State', 'value': 'AL'},
                        {'label': 'Total Diabetes Cases by State', 'value': 'DI'}
                    ],
                    value='TC'  # default value
                ),
              ],
              style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
              html.Div([
              dcc.Graph(id='fourth-graph')
            ], style={'width': '70%', 'display': 'inline-block'}),
            ], style={'display': 'flex', 'flex-direction': 'row', 'padding': '20px'}),

    # fifth dropdown and its graph
       html.Div([
        html.Div([
            html.P("Select from Dropdown 5:"),
            dcc.Dropdown(
                id='fifth-dropdown',
                options=[
                        {'label': 'Illinois', 'value': 'IL'},
                        {'label': 'Michigan', 'value': 'MI'},
                        {'label': 'Nevada', 'value': 'NV'},
                        {'label': 'Newjersey', 'value': 'NJ'},
                        {'label':'Newyork','value': 'NY'},
                        {'label': 'Total Cases of Alcohol in MI and NY ', 'value': 'AL_MI_NY'}
                    ],
                    value='IL'  # default value
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='fifth-graph')
        ], style={'width': '70%', 'display': 'inline-block'}),
    ], style={'display': 'flex', 'flex-direction': 'row','padding':'20px'}),
])

@app.callback(
    Output('third-graph', 'figure'),
    [Input('third-dropdown', 'value')]
)
def update_third_graph(selected_dropdown):
    if selected_dropdown == 'LOC':
        topic_of_interest = 'Cancer'
        locations = ['CA', 'UT', 'NV', 'AR']
        fig = go.Figure()
        for location in locations:
            filtered_data = visualization_df[(visualization_df['locationabbr'] == location) & (visualization_df['topic'] == topic_of_interest)]
            topic_cases_by_year_gender = filtered_data.groupby(['yearend', 'topic'])['datavalue'].sum().unstack(fill_value=0)
            for gender in topic_cases_by_year_gender.columns:
                fig.add_trace(go.Scatter(x=topic_cases_by_year_gender.index, y=topic_cases_by_year_gender[gender], mode='lines+markers', name=f'{location} - {gender}'))
        fig.update_layout(xaxis_title='Year', yaxis_title=f'Total Cases of {topic_of_interest}', title=f'Total Cases of {topic_of_interest} Over the Years by Location.', showlegend=True)

    elif selected_dropdown == 'MI':
        topics_of_interest = ['Asthma', 'Arthritis', 'Cancer', 'Alcohol']
        fig = go.Figure()
        for topic in topics_of_interest:
            filtered_data = visualization_df[(visualization_df['locationabbr'] == 'MI') & (visualization_df['topic'] == topic)]
            topic_cases_by_year_stratification = filtered_data.groupby(['yearend', 'topic'])['datavalue'].sum().unstack(fill_value=0)
            for stratification in topic_cases_by_year_stratification.columns:
                fig.add_trace(go.Scatter(x=topic_cases_by_year_stratification.index, y=topic_cases_by_year_stratification[stratification], mode='lines+markers', name=f'{topic} - {stratification}'))
        fig.update_layout(title='Total Cases of Topics Over Time in MI', xaxis_title='Year', yaxis_title='Total Cases', showlegend=True)

    elif selected_dropdown == 'NY':
        topics_of_interest = ['Cancer', 'Asthma', 'Alcohol', 'Arthritis']
        filtered_df = visualization_df[(visualization_df['locationabbr'] == 'NY') & (visualization_df['yearstart'] >= 2010) & (visualization_df['yearend'] <= 2021)]
        fig = go.Figure()
        for topic in topics_of_interest:
            topic_data = filtered_df[filtered_df['topic'] == topic]
            topic_data_grouped = topic_data.groupby('yearstart')['datavalue'].sum().reset_index()
            fig.add_trace(go.Scatter(x=topic_data_grouped['yearstart'], y=topic_data_grouped['datavalue'], mode='lines+markers', name=topic))
        fig.update_layout(title='Total Number of Cases per Topic Over Time in NY (2010-2021)', xaxis_title='Year', yaxis_title='Total Cases', showlegend=True)

    elif selected_dropdown == 'FL':
        topic_of_interest = 'Cancer'
        filtered_data = visualization_df[(visualization_df['locationabbr'] == 'FL') & (visualization_df['topic'] == topic_of_interest)]
        topic_cases_by_year_gender = filtered_data.groupby(['yearend', 'stratificationid1'])['datavalue'].sum().unstack(fill_value=0).reset_index()
        fig = px.line(topic_cases_by_year_gender, x='yearend', y=topic_cases_by_year_gender.columns[1:], labels={'value': 'Total Cases', 'variable': 'Gender'}, title=f'Total {topic_of_interest} Cases in FL Over the Years (Differentiated by Gender)')
        fig.update_layout(xaxis_title='Year', yaxis_title=f'Total {topic_of_interest} Cases', showlegend=True)

    elif selected_dropdown == 'AZ':
        topics_of_interest = ['Cardiovascular Disease', 'Chronic Kidney Disease', 'Disability', 'Diabetes']
        location_abbr = 'AZ'
        fig = go.Figure()
        for topic in topics_of_interest:
            filtered_data = visualization_df[(visualization_df['locationabbr'] == location_abbr) & (visualization_df['topic'] == topic)]
            topic_cases_by_year_stratification = filtered_data.groupby(['yearend', 'topic'])['datavalue'].sum().unstack(fill_value=0)
            for stratification in topic_cases_by_year_stratification.columns:
                fig.add_trace(go.Scatter(x=topic_cases_by_year_stratification.index, y=topic_cases_by_year_stratification[stratification], mode='lines+markers', name=f'{topic} - {stratification}'))
    return fig

@app.callback(
    Output('second-graph', 'figure'),
    [Input('second-dropdown', 'value')]
)
def update_second_graph(selected_dropdown):
    if selected_dropdown == 'NY':
        filtered_data = visualization_df[visualization_df['locationabbr'] == 'NY']
        topic_counts = filtered_data['topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        top_5_topics = topic_counts.head(5)
        fig = px.pie(top_5_topics,names='Topic',values='Count',title='Top 5 Topics in NY',)

    elif selected_dropdown == 'FL':
        fl_data = visualization_df[visualization_df['locationabbr'] == 'FL']
        top4_topics = fl_data['topic'].value_counts().head(4).index
        filtered_data = fl_data[fl_data['topic'].isin(top4_topics)]
        topic_cases_by_stratification = filtered_data.groupby(['topic', 'stratificationid1'])['datavalue'].sum().unstack(fill_value=0)
        fig = px.bar(topic_cases_by_stratification,x=topic_cases_by_stratification.index,y=['GENF', 'GENM'],title='Total Cases of Top 4 Topics by Gender in FL',labels={'GENF': 'Female', 'GENM': 'Male'},barmode='stack',)

    elif selected_dropdown == 'PA':
        diabetes_data_pa = visualization_df[(visualization_df['locationabbr'] == 'PA') &(df['topic'] == 'Diabetes') &(visualization_df['yearend'] >= 2015) &(visualization_df['yearend'] <= 2021) &(visualization_df['stratificationid1'].isin(['GENM', 'GENF']))]
        diabetes_cases_by_year = diabetes_data_pa.groupby(['yearend', 'stratificationid1'])['datavalue'].sum().reset_index()
        fig = px.bar(diabetes_cases_by_year,x='yearend',y='datavalue',color='stratificationid1',barmode='group',title='Diabetes Cases in PA by Year and Gender (2015-2021)',labels={'yearend': 'Year', 'datavalue': 'Total Cases', 'stratificationid1': 'Gender'})


    elif selected_dropdown == 'AR':
        ar_data = visualization_df[(visualization_df['locationabbr'] == 'AR') & (visualization_df['topic'] == 'Cardiovascular Disease')]
        topic_cases_by_year_gender = ar_data.groupby(['yearend', 'stratificationid1'])['datavalue'].sum().unstack(fill_value=0)
        fig = px.bar(topic_cases_by_year_gender,x=topic_cases_by_year_gender.index,y=['GENF', 'GENM'],title='Total Cardiovascular Disease Cases by Gender in AR Over the Years',labels={'GENF': 'Female', 'GENM': 'Male'},barmode='stack',)
        fig.update_xaxes(tickmode='array', tickvals=topic_cases_by_year_gender.index)

    elif selected_dropdown == 'TAR':
        ar_data = visualization_df[visualization_df['locationabbr'] == 'AR']
        top5_topics = ar_data['topic'].value_counts().head(5).index
        filtered_data = ar_data[ar_data['topic'].isin(top5_topics)]
        topic_cases_by_stratification = filtered_data.groupby(['topic', 'stratificationid1'])['datavalue'].sum().unstack(fill_value=0)
        fig = px.bar(topic_cases_by_stratification,x=topic_cases_by_stratification.index,y=topic_cases_by_stratification.columns,title='Total Cases of Top Topics by Stratification in AR',barmode='stack',labels={'value': 'Total Cases', 'variable': 'Stratification Category'})

    return fig


@app.callback(
Output('first-graph', 'figure'),
[Input('first-dropdown', 'value')]
)
def update_first_graph(selected_dropdown):

    if selected_dropdown == 'MDS':
        missing_data_summary = visualization_df.isnull().sum()
        fig = px.bar(missing_data_summary, x=missing_data_summary.index, y=missing_data_summary.values, labels={'x':'Column', 'y':'Missing Values'},title="Missing Data Summary")

    elif selected_dropdown == 'DTI':
        topicid_counts = visualization_df['topic'].value_counts().reset_index()
        topicid_counts.columns = ['Topic', 'Count']
        fig = px.pie(topicid_counts, values='Count', names='Topic', title='Distribution of Topic IDs')

    elif selected_dropdown == 'DS':
      datasource_counts = visualization_df['datasource'].value_counts().reset_index()
      datasource_counts.columns = ['Datasource', 'Count']
      top5_datasources = datasource_counts.head(5)
      top5_datasources['Ratio'] = top5_datasources['Count'] / top5_datasources['Count'].sum()
      fig = px.pie(top5_datasources, values='Ratio', names='Datasource', title='Top 5 Data Sources and Their Ratios')

    elif selected_dropdown == 'DVC':
      datavaluetype_counts = visualization_df['datavaluetype'].value_counts().reset_index()
      datavaluetype_counts.columns = ['datavaluetype', 'Count']
      top7_dvt = datavaluetype_counts.head(7)
      fig = px.pie(top7_dvt, names='datavaluetype', values='Count', title='Top 7 Datavaluetype by Count')

    elif selected_dropdown == 'SP':
      fig = px.scatter_3d(visualization_df, x='datavalue', y='highconfidencelimit', z='lowconfidencelimit',
                    title='3D Scatter Plot')
      fig.show()

    return fig

# Callback to update graph based on dropdown selection
@app.callback(
    Output('fourth-graph', 'figure'),
    [Input('fourth-dropdown', 'value')]
)
def update_fourth_graph(selected_disease):
    if selected_disease == 'TC':
        state_total_cases = visualization_df.groupby('locationabbr')['datavalue'].sum().reset_index()
        state_total_cases.columns = ['State', 'Total Cases']
        fig = px.choropleth(state_total_cases,locations="State",locationmode="USA-states",color="Total Cases",scope="usa",title="Total Cases per State in the USA",labels={"Total Cases": "Total Cases"},color_continuous_scale="Viridis" )

    elif selected_disease == 'AR':
        arthritis_totals = visualization_df[visualization_df['topic'] == 'Arthritis'].groupby('locationabbr')['datavalue'].sum().reset_index()
        arthritis_totals.rename(columns={'datavalue': 'TotalArthritisCases'}, inplace=True)
        fig = px.choropleth(arthritis_totals, locations="locationabbr", color="TotalArthritisCases", locationmode="USA-states", scope="usa", title="Total Arthritis Cases by State", labels={"TotalArthritisCases": "Total Arthritis Cases"})

    elif selected_disease == 'CA':
        cancer_total_cases = visualization_df[visualization_df['topic'] == 'Cancer'].groupby('locationabbr')['datavalue'].sum().reset_index()
        cancer_total_cases.rename(columns={'datavalue': 'TotalCancerCases'}, inplace=True)
        fig = px.choropleth(cancer_total_cases, locations="locationabbr", color="TotalCancerCases", locationmode="USA-states", scope="usa", title="Total Cancer Cases by State", labels={"TotalCancerCases": "Total Cancer Cases"})

    elif selected_disease == 'AL':
        alcohol_total_cases = visualization_df[visualization_df['topic'] == 'Alcohol'].groupby('locationabbr')['datavalue'].sum().reset_index()
        alcohol_total_cases.rename(columns={'datavalue': 'TotalAlcoholCases'}, inplace=True)
        fig = px.choropleth(alcohol_total_cases, locations="locationabbr", color="TotalAlcoholCases", locationmode="USA-states", scope="usa", title="Total Alcohol Cases by State", labels={"TotalAlcoholCases": "Total Alcohol Cases"})

    elif selected_disease == 'DI':
        diabetes_total_cases = visualization_df[visualization_df['topic'] == 'Diabetes'].groupby('locationabbr')['datavalue'].sum().reset_index()
        diabetes_total_cases.rename(columns={'datavalue': 'TotalDiabetesCases'}, inplace=True)
        fig = px.choropleth(diabetes_total_cases, locations="locationabbr", color="TotalDiabetesCases", locationmode="USA-states", scope="usa", title="Total Diabetes Cases by State", labels={"TotalDiabetesCases": "Total Diabetes Cases"})

    else:
        fig = go.Figure()  # Create an empty figure for any other cases or initial load

    return fig

# Callback to update graph based on dropdown selection
@app.callback(
    Output('fifth-graph', 'figure'),
    [Input('fifth-dropdown', 'value')]
)
def update_fifth_graph(selected_state):

    if selected_state in ['IL', 'NY', 'NJ', 'NV', 'MI']:
        state_data = visualization_df[visualization_df['locationabbr'] == selected_state]
        topic_counts = state_data['topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        topic_counts['Percentage'] = topic_counts['Count'] / topic_counts['Count'].sum() * 100
        fig = px.bar(topic_counts, x='Percentage', y='Topic', orientation='h', text='Percentage',
                     title=f'Distribution of Distinct Topics in {selected_state}')
        fig.update_traces(marker_color='lightgreen')
        fig.update_layout(xaxis_title='Percentage', yaxis_title='Topic', showlegend=False)

    elif selected_state == 'AL_MI_NY':

        filtered_data = visualization_df[(visualization_df['topic'] == 'Alcohol') & (visualization_df['locationabbr'].isin(['MI', 'NY']))]
        filtered_data = filtered_data[(filtered_data['yearend'] >= 2012) & (filtered_data['yearend'] <= 2020)]
        state_year_sums = filtered_data.groupby(['locationabbr', 'yearend'])['datavalue'].sum().reset_index()
        fig = px.bar(state_year_sums,x='locationabbr',y='datavalue',color='yearend',barmode='group',labels={'datavalue': 'Total Cases of Alcohol'},title='Total Cases of Alcohol in MI and NY from 2012 to 2020')
        fig.update_layout(xaxis_title='Location',yaxis_title='Total Cases of Alcohol',legend_title='Year')
    return fig

def prepare_data_and_train_model():
    # Load and preprocess data 
    prediction_df = pd.read_csv(url)
    prediction_df = prediction_df.drop('locationdesc', axis=1)
    prediction_df = prediction_df[prediction_df['datavaluetype'] == 'Number']
    prediction_df = prediction_df[prediction_df['topic'] == 'Cardiovascular Disease']
    prediction_df = prediction_df.drop(['datavalueunit', 'highconfidencelimit', 'lowconfidencelimit', 'questionid'], axis=1)

    # Encode categorical data for the prediction model
    local_label_encoder = LabelEncoder()
    prediction_df['location_encoded'] = local_label_encoder.fit_transform(prediction_df['locationabbr'])

    # Model training for the prediction model
    X = prediction_df[['yearstart', 'location_encoded']]
    y = prediction_df['datavalue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    local_xgb_model = XGBRegressor(random_state=42)
    local_xgb_model.fit(X_train, y_train)

    return local_xgb_model, local_label_encoder

# Prepare data and train model
xgb_model, label_encoder = prepare_data_and_train_model()


visualization_df = pd.read_csv(url)


state_codes = {
    'AL': '01', 'AK': '02', 'AZ': '03', 'AR': '04', 'CA': '05',
   'CO': '06', 'CT': '07', 'DE': '08', 'FL': '09', 'GA': '10',
    'HI': '11', 'ID': '12', 'IL': '13', 'IN': '14', 'IA': '15',
    'KS': '16', 'KY': '17', 'LA': '18', 'ME': '19', 'MD': '20',
    'MA': '21', 'MI': '22', 'MN': '23', 'MS': '24', 'MO': '25',
   'MT': '26', 'NE': '27', 'NV': '28', 'NH': '29', 'NJ': '30',
    'NM': '31', 'NY': '32', 'NC': '33', 'ND': '34', 'OH': '35',
    'OK': '36', 'OR': '37', 'PA': '38', 'RI': '39', 'SC': '40',
   'SD': '41', 'TN': '42', 'TX': '43', 'UT': '44', 'VT': '45',
   'VA': '46', 'WA': '47', 'WV': '48', 'WI': '49', 'WY': '50'
}
#Format state codes as a string for display
state_codes_str = ' | '.join(f"'{state}': '{code}'" for state, code in state_codes.items())

# Callback for making predictions
@app.callback(
    Output('prediction-result', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('input-state', 'value')]
)
def make_prediction(n_clicks, state_abbr):
    if n_clicks is None or not state_abbr:
        return 'Please enter the state abbreviation to get a prediction.'

    # Fixed year for all predictions
    year = 2023

    # Check if the state abbreviation is valid
    try:
        user_location_encoded = label_encoder.transform([state_abbr])
    except ValueError:
        
        return f"State code '{state_abbr}' is not recognized. Please enter a valid state code."

    # Make prediction using the model
    future_prediction_xgb = xgb_model.predict([[year, user_location_encoded[0]]])
    prediction = future_prediction_xgb[0]

    
    return f'Predicted value for {state_abbr} in {year}: {prediction}'

if __name__ == '__main__':
    app.run_server(debug=True)

# Import and use the predict_cardiovascular_disease_value function from predictor.py
from predictor import predict_cardiovascular_disease_value

state_code = 'CA'  
results = predict_cardiovascular_disease_value(state_code)

print(f"Predicted data value for {state_code}: {results['predicted_value']}")
print(f"RMSE: {results['rmse']}, MSE: {results['mse']}, R2 Score: {results['r2_score']}")
