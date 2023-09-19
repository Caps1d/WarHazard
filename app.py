import os

import dash
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

from modules import app_modules
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from dotenv import load_dotenv


load_dotenv()


# Load your data
fname = "Hazards_latest.csv"

df = pd.read_csv(app_modules.load_data(fname))
df = app_modules.format_data(df)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            app_modules.create_info_paragraphs()
                            + app_modules.add_input_location()
                            + app_modules.add_select_region_dropdown(df)
                            + app_modules.add_date_picker(df)
                            + app_modules.create_info_metrics()
                            + app_modules.create_markdown_description()
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flex-direction": "column",
                        "align-items": "right",
                        "padding-left": "8px",
                    },
                    className="desktopPadding",
                    xl=5,
                    lg=5,
                    md=12,
                    sm=12,
                    xs=12,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                html.Div(
                                    dcc.Graph(id="scatter-map", responsive=True),
                                    className="add_20_btm",
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                html.Div(
                                    app_modules.add_rangeslider(df),
                                    style={
                                        "margin-bottom": "20px",
                                        "position": "relative",
                                        "width": "100%",
                                    },
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                html.Div(
                                    app_modules.add_datatable(), className="add_20_btm"
                                ),
                                dbc.Col(app_modules.add_clear_btn(), width="auto"),
                            ]
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flex-direction": "column",
                        "align-items": "left",
                        "padding-left": "10px",
                        "padding-right": "10px",
                    },
                    xl=7,
                    lg=7,
                    md=12,
                    sm=12,
                    xs=12,
                ),
            ],
            style={"padding-top": "40px"},
        ),
        dbc.Row(
            [
                html.Div(
                    [
                        html.P("Project Overview", className="title"),
                    ]
                ),
                app_modules.create_markdown_overview(),
            ],
            style={
                "justify-content": "center",
                "padding-top": "40px",
            },
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P("Methodology", className="title"),
                    ]
                ),
            ],
            style={
                "padding-top": "40px",
                "padding-bottom": "6px",
            },
        ),
        app_modules.create_markdown_methodology(),
        html.Script(
            """
        window.onload = function() {
            window.dispatchEvent(new Event('resize'));
        };
    """
        ),
        app_modules.create_markdown_methodology_bottom(),
        app_modules.add_author(),
    ],
    fluid=True,
)


mapbox_access_token = os.getenv("MAPBOX_ACCESS_TOKEN")
custom_mapbox_style_url = os.getenv("MAPBOX_STYLE")


@app.callback(
    Output("scatter-map", "figure"),
    Output("total-events", "children"),
    Output("selected-events", "children"),
    Output("hazard-range", "children"),
    [
        Input("search-bar", "value"),
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("hazard-score-slider", "value"),
        Input("region-dropdown", "value"),
    ],
)

# map function
def update_map_and_events(location, start_date, end_date, hazard_score, region):
    # If alocation is provided, filter the dataframe to that location
    if location:
        dff = df[df["location"] == location]
    else:
        dff = df

    # Filter by date range
    dff = dff[(dff["event_date"] >= start_date) & (dff["event_date"] <= end_date)]

    # Filter by hazard score range
    dff = dff[
        (dff["hazard_score"] >= hazard_score[0])
        & (dff["hazard_score"] <= hazard_score[1])
    ]

    # Filter by region
    if region:
        dff = dff[dff["admin2"].isin(region)]

    # Here, group the dataframe by 'location' and select the row with the max 'event_date' for each group
    dff = dff.loc[dff.groupby("location")["event_date"].idxmax()]

    # Define your color scheme
    color_scale = [[0, "darkgreen"], [0.5, "yellow"], [1.0, "red"]]

    # Create the scatter mapbox figure using go.Scattermapbox
    dff["hover_text"] = (
        "Location: "
        + dff["location"].astype(str)
        + "<br>Hazard Score: "
        + dff["hazard_score"].astype(str)
        + "<br>Date: "
        + dff["event_date"].astype(str)
    )
    total_events = str(len(df))
    selected_events = str(len(dff))
    hazard_range = str(hazard_score[0]) + " - " + str(hazard_score[1])

    fig = go.Figure(
        data=go.Scattermapbox(
            lat=dff["latitude"],
            lon=dff["longitude"],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=7,  # Adjust size as needed
                color=dff["hazard_score"],
                colorscale=color_scale,
                opacity=dff["opacity"],  # set the opacity here
            ),
            hovertext=dff["hover_text"],
            hoverinfo="text",
        )
    )

    # Update the map layout
    fig.update_layout(
        mapbox=dict(
            accesstoken=mapbox_access_token,
            style=custom_mapbox_style_url,
            bearing=0,
            pitch=0,
            zoom=5,
            center=dict(lat=48.3794, lon=31.1656),  # Center of the map on Ukraine
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},  # Remove the margins
        legend=dict(bgcolor="#111111", font=dict(color="#7FDBFF")),
    )

    return fig, total_events, selected_events, hazard_range


@app.callback(
    Output("table", "data"),
    [Input("scatter-map", "clickData"), Input("close-button", "n_clicks")],
)
def update_table(clickData, n_clicks):
    ctx = dash.callback_context

    # if callback was triggered by scatter-map clickData
    if ctx.triggered[0]["prop_id"] == "scatter-map.clickData":
        if clickData:
            clicked_location = (
                clickData["points"][0]["hovertext"].split("<br>")[0].split(": ")[1]
            )
            dff = df[df["location"] == clicked_location]
            dff["event_date"] = dff["event_date"].dt.strftime(
                "%Y-%m-%d"
            )  # Format the event_date column
            return dff.to_dict("records")

    # if callback was triggered by close-button n_clicks
    # elif ctx.triggered[0]["prop_id"] == "close-button.n_clicks":
    #     return []  # return an empty list when the close button is clicked
    #
    else:
        return []  # return an empty list when no marker is clicked


# Run the app
if __name__ == "__main__":
    app.run_server(debug=False)
