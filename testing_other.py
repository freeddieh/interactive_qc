import io
import dash
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State

### Made in 2024 by Frederik Hildebrand ###

# Placeholder data before upload
data = {
    'x': [np.nan],
    'y': [np.nan],
    'Flag': [np.nan]
}

df = pd.DataFrame(data)

# Initialize the Dash app
app = dash.Dash(__name__)

# Create Plotly scatter plot
def create_figure(df):
    fig = go.Figure()
    
    # Add data with customdata initialized (set value as customdata for each point)
    fig.add_trace(go.Scattergl(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['Flag'],  # Use value as color scale
            colorscale='RdYlGn',  # Color scale to show values
            showscale=True
        ),
        text=(df['Flag']),  # Tooltip text will show value
        hoverinfo='text',
        customdata=df['Flag'],  # Initialize customdata with the values
        selected=dict(marker=dict(color='red', size=9))  # Style for selected points
    ))

    fig.update_layout(
        title='Interactive Data Flagging',
        clickmode='event+select',  # Enable click interaction
        hovermode='closest',
        dragmode='select'  # Enable selection of multiple points
    )
    return fig

# Layout for the Dash app
app.layout = html.Div([
    # Graph
    html.H1('Graph and flag data'),
    dcc.Graph(id='scatter-plot', figure=create_figure(df)),

    dbc.Row([
        # Column 1: Upload File
        dbc.Col([
            html.H4('Upload .csv or .txt File', style={'textAlign': 'center'}),
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload File', id='upload-button', n_clicks=0),
                multiple=False,
                style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center'}
            ),
        ], width=4, style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center'}),  # Center the column contents

        # Column 2: Input new data flag
        dbc.Col([
            html.H4('Enter new data flag for the selected data', style={'textAlign': 'center'}),
            dcc.Input(id='value-input', type='number', debounce=True, style={'textAlign': 'center'}),
            html.Button('Update Flag', id='submit-button', n_clicks=0, style={'display': 'block', 'margin': '0 auto'}),
        ], width=4, style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center'}),  # Center the column contents

        # Column 3: Input new file name and save button
        dbc.Col([
            html.H4('Enter name for .csv file of data with updated flags', style={'textAlign': 'center'}),
            dcc.Input(id='file-name', type='text', debounce=True, style={'textAlign': 'center'}),
            html.Button('Save Updated Data', id='save-button', n_clicks=0, style={'display': 'block', 'margin': '0 auto'}),
        ], width=4, style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center'}),  # Center the column contents
    ]),

    # Store components to track intermediate data states
    dcc.Store(id='filtered-data-store', data=df.to_dict('records')),  # Stores filtered data for table
    dcc.Store(id='data-store', data=df.to_dict('records')),  # Store the full dataset
    dcc.Store(id='button1-state', data=0),  # Tracks button state to prevent repeated updates
    dcc.Store(id='button2-state', data=0),  # Tracks button state to prevent repeated updates
    dcc.Store(id='button3-state', data=0),  # Tracks button state to prevent repeated updates
    dcc.Store(id='last-x-range', data=None),  # Track last valid x-axis range
    dcc.Store(id='last-y-range', data=None),  # Track last valid y-axis range

    ## Save the data with updated data flags saving
    #html.Div([

    #]),  
    #dcc.Store(id='button-state', data=0),  # Tracks button state to prevent repeated updates

    # DataTable to display the DataFrame (without the Index column)
    dash_table.DataTable(
        id='data-table',
        columns=[
            {'name': 'Date', 'id': 'x'},
            {'name': 'Data', 'id': 'y'},
            {'name': 'Flag', 'id': 'Flag'},
        ],
        data=df.to_dict('records'),  # Initialize table with the data
        style_table={'marginTop': '20px'},  # Styling to separate table from graph
        style_cell={'textAlign': 'center'},  # Center-align cell text
        style_header={'backgroundColor': 'lightgrey'},  # Style for the header
    ),
    
    # Hidden Div to track the axis range for filtering the table
    html.Div(id='axis-ranges', style={'display': 'none'})
])




#-----------------------------------------------------------------------------#
##############################   Callback Start   #############################
#-----------------------------------------------------------------------------# 

# Callback to handle point click and input value
@app.callback(
    [Output('scatter-plot', 'figure'), # Scatterplot of the data
     Output('button1-state', 'data'), # Total number of button clicks for Flag update
     Output('button2-state', 'data'), # Total number of button clicks for Save update
     Output('button3-state', 'data'), # Total number of button clicks for Upload update
     Output('data-table', 'data'), # 
     Output('scatter-plot', 'selectedData'),  # Selected data in the plot update
     Output('axis-ranges', 'children'), # x- and y-axis ranges
     Output('filtered-data-store', 'data'), # Store of the filtered data for table
     Output('last-x-range', 'data'),  # Store the last valid x-axis range
     Output('last-y-range', 'data')],  # Store the last valid y-axis range
    [Input('upload-data', 'contents'),
     Input('scatter-plot', 'selectedData'),
     Input('submit-button', 'n_clicks'),
     Input('save-button', 'n_clicks'),
     Input('upload-button', 'n_clicks'),
     Input('scatter-plot', 'relayoutData'), # Listen to axis range updates
     Input('data-store', 'data')], # Data store for the unfiltered dataset 
    [State('value-input', 'value'),
     State('file-name', 'value'),
     State('scatter-plot', 'figure'),
     State('button1-state', 'data'),
     State('button2-state', 'data'),
     State('button3-state', 'data'),
     State('last-x-range', 'data'),  # Get the last valid x-range
     State('last-y-range', 'data')]  # Get the last valid y-range
)
def update_plot(contents,      # Tracks selected file
                selectedData,  # Tracks selected data in the plot
                n_clicks,      # Tracks numbers of clicks of the update button
                save_n_clicks, # Tracks numbers of clicks of the save button
                data_n_clicks, # Tracks numbers of clicks of the upload button
                relayout_data, # Tracks plot axes
                stored_data,   ### Stores the entire dataset
                new_value,     # New flag input
                savefile_name, # Name of .csv savefile
                figure,        # Plot
                button1_state, # Update button state
                button2_state, # Save button state
                button3_state, # Upload button state
                last_x_range,  # Previous range for the x-axis
                last_y_range   # Previous range for the y-axis
                ):
    # Convert the stored_data back into a Pandas DataFrame 
    # for filtering based on the x-ranges 
    df_full = pd.DataFrame(stored_data)

    # If figure is None, initialize df and customdata
    if figure is None:
        figure = create_figure(df)

    # Ensure that df is properly updated from the figure
    df = pd.DataFrame({
        'x': figure['data'][0]['x'],
        'y': figure['data'][0]['y'],
        'Flag': figure['data'][0]['customdata']
    })

    # If figure is None, initialize df and customdata
    if data_n_clicks > button3_state and contents is not None:
        # Extract file content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Check if the file is a CSV or TXT file
            if 'csv' in content_type:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            else:
                # Try reading as a text file and assuming it's tabular
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t')

            # Ensure the timestamp column is a datetime object
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Resample data for zoomed-out views (e.g., daily aggregation)
            df_resampled = df.resample('D', on='timestamp').mean().reset_index(names='timestamp')
            df_resampled['timestamp'] = pd.to_datetime(df_resampled['timestamp'])

            figure = create_figure(df)
            button3_state = data_n_clicks  # Set the button state to current n_clicks value

        except Exception as e:
            raise Exception(f'An Error occured when loading the data ({e}).\nPlease ensure the file format is .csv or .txt.')

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
    #print(f'X-axis range: {x_range}, Y-axis range: {y_range}')

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
    if n_clicks > button1_state and selected_indices and new_value is not None:
        # Update the value for each selected point
        for idx in selected_indices:
            df.loc[idx, 'Flag'] = new_value

        # Update the figure's data
        figure['data'][0]['customdata'] = df['Flag'].tolist()  # Update customdata
        for idx in selected_indices:
            figure['data'][0]['text'][idx] = new_value  # Update the text shown on hover
            figure['data'][0]['marker']['color'][idx] = new_value  # Update color

        # Reset the button state to prevent further updates until next click
        button1_state = n_clicks  # Set the button state to current n_clicks value

        # Clear the selection box after update
        return figure, button1_state, button2_state, button3_state, df_filtered_top100[['x', 'y', 'Flag']].to_dict('records'), None, \
               f'X-axis range: {x_range} | Y-axis range: {y_range}', df_filtered_top100.to_dict('records'), last_x_range, last_y_range

    if save_n_clicks > button2_state and savefile_name is not None:
        df.to_csv(f'{str(savefile_name)}.csv')
        button2_state = save_n_clicks  # Set the button state to current n_clicks value

        return figure, button1_state, button2_state, button3_state, df_filtered_top100[['x', 'y', 'Flag']].to_dict('records'), None, \
               f'X-axis range: {x_range} | Y-axis range: {y_range}', df_filtered_top100.to_dict('records'), last_x_range, last_y_range

    # If no update is performed, just return the figure and table data without changes
    return figure, button1_state, button2_state, button3_state, df_filtered_top100[['x', 'y', 'Flag']].to_dict('records'), selectedData, \
           f'X-axis range: {x_range} | Y-axis range: {y_range}', df_filtered_top100.to_dict('records'), last_x_range, last_y_range

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)