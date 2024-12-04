import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Sample data
data = pd.DataFrame({
    'x': np.linspace(0, 10, 100),
    'y': np.sin(np.linspace(0, 10, 100)),
    'Flag': np.random.randint(1, 4, 100)
})

# data2 = pd.read_csv(')

df = pd.DataFrame(data)

# Initialize the Dash app
app = dash.Dash(__name__)

# Create Plotly scatter plot
def create_figure(df):
    fig = go.Figure()
    
    # Add data with customdata initialized (set value as customdata for each point)
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(
            size=12,
            color=df['Flag'],  # Use value as color scale
            colorscale='RdYlGn',  # Color scale to show values
            showscale=True
        ),
        text=df['Flag'],  # Tooltip text will show value
        hoverinfo="text",
        customdata=df['Flag'],  # Initialize customdata with the values
        selected=dict(marker=dict(color='red', size=14))  # Style for selected points
    ))

    fig.update_layout(
        title="Interactive Data Flagging",
        clickmode="event+select",  # Enable click interaction
        hovermode="closest",
        dragmode='select'  # Enable selection of multiple points
    )
    return fig

# Layout for the Dash app
app.layout = html.Div([
    # Graph
    dcc.Graph(id='scatter-plot', figure=create_figure(df)),

    # Input field and submit button
    html.Div([
        html.Label("Enter new data flag for the selected data:"),
        dcc.Input(id='value-input', type='number', debounce=True),
        html.Button('Update Flag', id='submit-button', n_clicks=0),
    ]),

    # Store components to track intermediate data states
    dcc.Store(id='filtered-data-store', data=df.to_dict('records')),  # Stores filtered data for table
    dcc.Store(id='button-state', data=0),  # Tracks button state to prevent repeated updates
    dcc.Store(id='last-x-range', data=None),  # Track last valid x-axis range
    dcc.Store(id='last-y-range', data=None),  # Track last valid y-axis range

    # DataTable to display the DataFrame (without the Index column)
    dash_table.DataTable(
        id='data-table',
        columns=[
            {"name": "X", "id": "x"},
            {"name": "Y", "id": "y"},
            {"name": "Flag", "id": "Flag"},
        ],
        data=df.to_dict('records'),  # Initialize table with the data
        style_table={'marginTop': '20px'},  # Styling to separate table from graph
        style_cell={'textAlign': 'center'},  # Center-align cell text
        style_header={'backgroundColor': 'lightgrey'},  # Style for the header
    ),
    
    # Hidden Div to track the axis range for filtering the table
    html.Div(id='axis-ranges', style={'display': 'none'})
])

# Callback to handle point click and input value
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('button-state', 'data'),
     Output('data-table', 'data'),
     Output('scatter-plot', 'selectedData'),  # Clear selectedData after update
     Output('axis-ranges', 'children'),
     Output('filtered-data-store', 'data'),
     Output('last-x-range', 'data'),  # Store the last valid x-axis range
     Output('last-y-range', 'data')],  # Store the last valid y-axis range
    [Input('scatter-plot', 'selectedData'),
     Input('submit-button', 'n_clicks'),
     Input('scatter-plot', 'relayoutData')],  # Listen to axis range updates
    [State('value-input', 'value'),
     State('scatter-plot', 'figure'),
     State('button-state', 'data'),
     State('last-x-range', 'data'),  # Get the last valid x-range
     State('last-y-range', 'data')]  # Get the last valid y-range
)
def update_plot(selectedData, n_clicks, relayout_data, new_value, figure, button_state, last_x_range, last_y_range):

    # If figure is None, initialize df and customdata
    if figure is None:
        figure = create_figure(df)
    
    # Ensure that df is properly updated from the figure
    df = pd.DataFrame({
        'x': figure['data'][0]['x'],
        'y': figure['data'][0]['y'],
        'Flag': figure['data'][0]['customdata']
    })

    # Initialize the filtered dataframe
    df_filtered = df

    # Get the current axis ranges from the relayoutData
    if relayout_data is None or next((True for axrng in relayout_data if 'autorange' in axrng), False):
        if relayout_data is None or next((True for axrng in relayout_data if 'xaxis' in axrng), False) and \
         next((True for axrng in relayout_data if 'yaxis' in axrng), False):
            x_range = (None, None)
            y_range = (None, None)
            df_filtered = df  # Reset to full dataset when both axes are auto-ranged
            last_x_range = (None, None)
            last_y_range = (None, None)
        else:
            # If only one axis is auto-ranged, preserve the other axis range and use the last valid range
            if next((True for axrng in relayout_data if 'xaxis' in axrng), False):
                x_range = (None, None)  # Reset x-axis
                y_range = last_y_range  # Keep the last valid y-range
            elif next((True for axrng in relayout_data if 'yaxis' in axrng), False):
                y_range = (None, None)  # Reset y-axis
                x_range = last_x_range  # Keep the last valid x-range
    else:
        x_range = (relayout_data.get('xaxis.range[0]', None), relayout_data.get('xaxis.range[1]', None))
        y_range = (relayout_data.get('yaxis.range[0]', None), relayout_data.get('yaxis.range[1]', None))

    # Debugging: Print x_range and y_range to track behavior
    #print(f"X-axis range: {x_range}, Y-axis range: {y_range}")

    # Track the last valid x-range if available
    if all(ele is not None for ele in x_range):
        last_x_range = x_range
    elif (last_x_range is not None and relayout_data is not None) and next((False for axrng in relayout_data if 'xaxis.autorange' in axrng), True):
        x_range = last_x_range  # Use the previous valid x-range if available

    # Track the last valid y-range if available
    if all(ele is not None for ele in y_range):
        last_y_range = y_range
    elif (last_y_range is not None and relayout_data is not None) and next((False for axrng in relayout_data if 'yaxis.autorange' in axrng), True):
        y_range = last_y_range  # Use the previous valid y-range if available

    # Apply filtering based on the current x and y axis ranges
    if all(ele is not None for ele in x_range):
        # Ensure valid x-range and handle potential floating-point precision issues
        df_filtered = df_filtered[(df_filtered['x'] >= x_range[0]) & (df_filtered['x'] <= x_range[1])]
    
    if all(ele is not None for ele in y_range):
        df_filtered = df_filtered[(df_filtered['y'] >= y_range[0]) & (df_filtered['y'] <= y_range[1])]

    # Show only the 100 first values of the sliced DataFrame
    df_filtered_top100 = df_filtered.head(100)

    # Handle multiple point selection (from selectedData)
    if selectedData:
        selected_indices = [point['pointIndex'] for point in selectedData['points']]
    else:
        selected_indices = []

    # Handle the Update Value logic when button is clicked
    if n_clicks > button_state and selected_indices and new_value is not None:
        # Update the value for each selected point
        for idx in selected_indices:
            df.loc[idx, 'Flag'] = new_value

        # Update the figure's data
        figure['data'][0]['customdata'] = df['Flag'].tolist()  # Update customdata
        for idx in selected_indices:
            figure['data'][0]['text'][idx] = new_value  # Update the text shown on hover
            figure['data'][0]['marker']['color'][idx] = new_value  # Update color

        # Reset the button state to prevent further updates until next click
        button_state = n_clicks  # Set the button state to current n_clicks value

        # Clear the selection box after update
        return figure, button_state, df_filtered_top100[['x', 'y', 'Flag']].to_dict('records'), None, \
               f"X-axis range: {x_range} | Y-axis range: {y_range}", df_filtered_top100.to_dict('records'), last_x_range, last_y_range

    # If no update is performed, just return the figure and table data without changes
    return figure, button_state, df_filtered_top100[['x', 'y', 'Flag']].to_dict('records'), selectedData, \
           f"X-axis range: {x_range} | Y-axis range: {y_range}", df_filtered_top100.to_dict('records'), last_x_range, last_y_range

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)