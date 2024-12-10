import dash
from dash import Input, Output, dcc, html
import pandas as pd
import plotly.graph_objs as go
import numpy as np

date_range = pd.date_range('2022-01-01', periods=300000, freq='min')
df = pd.DataFrame({
    'timestamp': date_range,
    'value': np.random.random(300000)
})

# Ensure the timestamp column is a datetime object
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Resample data for zoomed-out views (e.g., daily aggregation)
df_resampled = df.resample('D', on='timestamp').mean().reset_index(names='timestamp')
df_resampled['timestamp'] = pd.to_datetime(df_resampled['timestamp'])

# Start the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='scatter-plot'),
    dcc.Store(id='data-store', data=df.to_dict('records'))  # Store the full dataset
])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-plot', 'relayoutData'),
     Input('scatter-plot', 'selectedData'),
     Input('data-store', 'data')]
)
def update_scatterplot(relayoutData, selectedData, stored_data):
    # Convert stored_data back to DataFrame for filtering
    df_full = pd.DataFrame(stored_data)

    # Initialize the figure with default data (entire dataset) or empty figure
    if relayoutData is None or 'xaxis.range' not in relayoutData:
        return go.Figure(data=go.Scattergl(
            x=df_full['timestamp'],
            y=df_full['value'],
            mode='markers',
            marker=dict(size=5)
        ))

    # Extract the x-axis range from relayoutData
    xaxis_range = relayoutData.get('xaxis.range', None)
    if not xaxis_range:
        return go.Figure()

    # Convert the start and end of the range to datetime
    start, end = xaxis_range
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    # Filter data based on the zoom level
    df_filtered = df_full[(df_full['timestamp'] >= start) & (df_full['timestamp'] <= end)]

    # If no data is available in the range, use the full dataset
    if df_filtered.empty:
        print("No data in the zoom range. Rendering full dataset instead.")
        df_filtered = df_full

    # Create the figure with scattergl for better performance with large datasets
    fig = go.Figure(data=go.Scattergl(
        x=df_filtered['timestamp'],
        y=df_filtered['value'],
        mode='markers',
        marker=dict(size=5)
    ))

    # Handle point selection (optional)
    if selectedData is not None:
        selected_points = selectedData['points']
        print("Selected Points:", selected_points)

        # Optional: Highlight selected points by changing their appearance
        selected_x = [point['x'] for point in selected_points]
        selected_y = [point['y'] for point in selected_points]

        # Update marker appearance for selected points
        fig.update_traces(marker=dict(color='red', size=10), selector=dict(x=selected_x, y=selected_y))

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    
    
    
def create_large_dataset_figure(
    df,
    colorscale="RdYlGn",
    marker_size=5,
    max_points=10000,
    show_color_scale=True
):
    """
    Create a high-performance interactive scatter plot for large datasets.

    Parameters:
        df (pd.DataFrame): DataFrame containing columns 'x', 'y', and 'Flag'.
        colorscale (str): Color scale for the markers.
        marker_size (int): Size of the scatter plot markers.
        max_points (int): Maximum number of points to display for performance.
        show_color_scale (bool): Whether to show the color scale on the plot.

    Returns:
        plotly.graph_objects.Figure: The optimized scatter plot.
    """
    # Downsample the data if it exceeds the maximum number of points
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)

    # Normalize the 'Flag' column for efficient color mapping
    flag_min, flag_max = df['Flag'].min(), df['Flag'].max()
    df['Flag_normalized'] = (df['Flag'] - flag_min) / (flag_max - flag_min)

    # Create the figure
    fig = go.Figure()

    # Add the Scattergl trace for high-performance rendering
    fig.add_trace(go.Scattergl(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(
            size=marker_size,
            color=df['Flag_normalized'],  # Use normalized values for color
            colorscale=colorscale,
            showscale=show_color_scale
        ),
        customdata=df[['Flag']].values,  # Include only necessary data
        hovertemplate=(
            "Flag: %{customdata[0]}<br>"
            "X: %{x:.2f}<br>"
            "Y: %{y:.2f}<extra></extra>"
        ),  # Custom hover template
        selected=dict(marker=dict(color='red', size=marker_size + 2))  # Highlight selected points
    ))

    # Update layout for interactivity and performance
    fig.update_layout(
        title="High-Performance Scatter Plot",
        clickmode="event+select",  # Enable click and select interactions
        hovermode="closest",  # Highlight the closest point on hover
        dragmode='select',  # Enable box/lasso selection
        uirevision="dataset"  # Prevent resetting when updating the figure
    )

    return fig