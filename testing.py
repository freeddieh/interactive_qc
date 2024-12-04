import io
import dash
import base64
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State

# Placeholder data before upload
data = {
    'x': [np.nan],
    'y': [np.nan],
    'Flag': [np.nan]
}

df = pd.DataFrame(data)

xs, ys, flags = ['x'], ['y'], ['Flag']

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create Plotly scatter plot
def create_figure(df, xs, ys, flags):
    if all(inp is not None for inp in (xs, ys, flags)):
        # Placeholder data before upload
        data = {
            'x': [np.nan],
            'y': [np.nan],
            'Flag': [np.nan]
        }

        df = pd.DataFrame(data)

        xs, ys, flags = ['x'], ['y'], ['Flag']

    fig = go.Figure()
    
    # Add data with customdata initialized (set value as customdata for each point)
    for idx, y_col in enumerate(ys):
        fig.add_trace(go.Scatter(
            x=df[xs],
            y=df[y_col],
            mode='markers',
            marker=dict(
                size=12,
                color=df[flags[idx]],  # Use value as color scale
                colorscale='RdYlGn',  # Color scale to show values
                showscale=True
            ),
            text=(df[flags]),  # Tooltip text will show value
            hoverinfo='text',
            customdata=df[flags],  # Initialize customdata with the values
            selected=dict(marker=dict(color='red', size=14))  # Style for selected points
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

    dbc.Row([
        dbc.Col([
            html.H4('Select X-Column', style={'textAlign': 'center'}),
            dcc.Dropdown(
                id='X-column-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns],
                multi=False,  # Enable multiple selection
                placeholder='Select X-Column'
            ), 
        ], style={'textAlign': 'left', 'width': '30%'}),  # This will be 1/3 of the row

        dbc.Col([
            html.H4('Select Y-Column(s)', style={'textAlign': 'center'}),
            dcc.Dropdown(
                id='Y-column-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns],
                multi=True,  # Enable multiple selection
                placeholder='Select Y-Column(s)'
            ),
        ], style={'textAlign': 'left', 'width': '30%'}),  # This will be 1/3 of the row

        dbc.Col([
            html.H4('Select Flag-Column', style={'textAlign': 'center'}),
            dcc.Dropdown(
                id='F-column-dropdown',
                options=[{'label': col, 'value': col} for col in df.columns],
                multi=True,  # Enable multiple selection
                placeholder='Select Flagging Column(s)'
            ),
        ], style={'textAlign': 'left', 'width': '30%'}),  # This will be 1/3 of the row
    ], justify='center'),

    dcc.Graph(id='scatter-plot', figure=create_figure(df, xs, ys, flags)),

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
    dcc.Store(id='button1-state', data=0),  # Tracks button state to prevent repeated updates
    dcc.Store(id='button2-state', data=0),  # Tracks button state to prevent repeated updates
    dcc.Store(id='button3-state', data=0),  # Tracks button state to prevent repeated updates
    dcc.Store(id='last-x-range', data=None),  # Track last valid x-axis range
    dcc.Store(id='last-y-range', data=None),  # Track last valid y-axis range

    # DataTable to display the DataFrame (without the Index column)

    ### Try to insert the Flag update input in the column header for the table, so it is easier to correlate Flags and ###
    ### their corresponding values
    dash_table.DataTable(
        id='data-table',
        columns=[],  # Initially empty, will be populated by callback
        data=df.to_dict('records'),  # Initialize table with the data
        style_table={'marginTop': '20px', 'height': '350px', 'overflowY': 'auto'},  # Styling to separate table from graph
        style_header={'backgroundColor': 'lightgrey'},  # Style for the header
        style_cell={'textAlign': 'center'},  # Center cell content
    ),
    
    # Hidden Div to track the axis range for filtering the table
    html.Div(id='axis-ranges', style={'display': 'none'})
])

#------------------------------------------------------------------------------#
##############################   Callback Start   ##############################
#------------------------------------------------------------------------------# 

# Callback to handle point click and input value
@app.callback(
    [Output('X-column-dropdown', 'value'),
     Output('Y-column-dropdown', 'value'),
     Output('F-column-dropdown', 'value'),
     Output('scatter-plot', 'figure'),
     Output('button1-state', 'data'),
     Output('button2-state', 'data'),
     Output('button3-state', 'data'),
     Output('data-table', 'data'),
     Output('data-table', 'columns'),
     Output('scatter-plot', 'selectedData'),  # Clear selectedData after update
     Output('axis-ranges', 'children'),
     Output('filtered-data-store', 'data'),
     Output('last-x-range', 'data'),  # Store the last valid x-axis range
     Output('last-y-range', 'data')],  # Store the last valid y-axis range
    [Input('upload-data', 'contents'),
     Input('X-column-dropdown', 'value'),
     Input('Y-column-dropdown', 'value'),
     Input('F-column-dropdown', 'value'),
     Input('scatter-plot', 'selectedData'),
     Input('submit-button', 'n_clicks'),
     Input('save-button', 'n_clicks'),
     Input('upload-button', 'n_clicks'),
     Input('scatter-plot', 'relayoutData')],  # Listen to axis range updates
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
                xs,            # Selected x-columns
                ys,            # Selected y-columns
                flags,         # Selected flag-column
                selectedData,  # Tracks selected data in the plot
                n_clicks,      # Tracks numbers of clicks of the update button
                save_n_clicks, # Tracks numbers of clicks of the save button
                data_n_clicks, # Tracks numbers of clicks of the upload button
                relayout_data, # Tracks plot axes
                new_value,     # New flag input
                savefile_name, # Name of .csv savefile
                figure,        # Plot
                button1_state, # Update button state
                button2_state, # Save button state
                button3_state, # Upload button state
                last_x_range,  # Previous range for the x-axis
                last_y_range   # Previous range for the y-axis
                ):

    # If figure is None, initialize df and customdata
    if figure is None:
        figure = create_figure(df, xs, ys, flags)

    # Sort the selected columns according to their order in the DataFrame
    if xs is not None:
        sorted_xs = sorted(xs, key=lambda col: df.columns.get_loc(col))
    else: 
        sorted_xs = []
        
    if ys is not None: 
        sorted_ys = sorted(xs, key=lambda col: df.columns.get_loc(col))
    else:
        sorted_ys = []

    if flags is not None:
        sorted_flags = sorted(xs, key=lambda col: df.columns.get_loc(col))
    else:
        sorted_flags = []

    # Update the table columns and data
    all_columns = [{'name': col, 'id': col} for col in sorted(list(sorted_xs +
                                                              sorted_ys +
                                                              sorted_flags),
                                                              key=lambda col: df.columns.get_loc(col))]  # Define columns for the table

    # Ensure that df is properly updated from the figure
    data = {
        'x': figure['data'][0]['x'],
        'y': figure['data'][0]['y'],
        'Flag': figure['data'][0]['customdata']
    }

    df = pd.DataFrame(data)

    # Initialize the filtered dataframe
    df_filtered = df
    print(df_filtered.head())

    # If figure is None, initialize df and customdata
    if data_n_clicks > button3_state and contents is not None:
        # Extract file content
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            # Check if the file is a .csv or .txt file
            if 'csv' in content_type:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            else:
                # Try reading as a text file and assuming it is tabular
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep='\t')

            figure = create_figure(df, xs, ys, flags)
            button3_state = data_n_clicks  # Set the button state to current n_clicks value

        except Exception as e:
            raise Exception(f'An Error occured when loading the data ({e}).\nPlease ensure the file format is .csv or .txt.')
    
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

    if not all_columns:
        df_filtered_top100 = []
        df_filtered_sorted = []
    else:
        # Show only the 100 first values of the sliced DataFrame
        df_filtered_top100 = df_filtered.head(100).to_dict('records')
        df_filtered_sorted = df_filtered_top100[[list(col.values())[0] for col in all_columns]].to_dict('records')


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
        return sorted_xs, sorted_ys, sorted_flags, figure, button1_state, button2_state, \
               button3_state, df_filtered_sorted, all_columns, None,\
               f'X-axis range: {x_range} | Y-axis range: {y_range}', df_filtered_top100, last_x_range, last_y_range

    if save_n_clicks > button2_state and savefile_name is not None:
        df.to_csv(f'{str(savefile_name)}.csv')
        button2_state = save_n_clicks  # Set the button state to current n_clicks value

        return sorted_xs, sorted_ys, sorted_flags, figure, button1_state, button2_state, \
               button3_state, df_filtered_sorted, all_columns, None,\
               f'X-axis range: {x_range} | Y-axis range: {y_range}', df_filtered_top100, last_x_range, last_y_range

    # If no update is performed, just return the figure and table data without changes
    return sorted_xs, sorted_ys, sorted_flags, figure, button1_state, button2_state, \
           button3_state, df_filtered_sorted, all_columns, selectedData, \
           f'X-axis range: {x_range} | Y-axis range: {y_range}', df_filtered_top100, last_x_range, last_y_range

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)