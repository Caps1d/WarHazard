import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import json
import os
from dotenv import load_dotenv

load_dotenv()

dir_path = os.path.dirname(os.path.realpath(__file__))
commands_dir = os.path.join(dir_path, "hazard_scores", "Hazards_Jan_July.csv")


# Load your data
df = pd.read_csv(commands_dir)
df["event_date"] = pd.to_datetime(df["event_date"])

df["unique_id"] = df["admin2"] + "-" + df["admin3"] + "-" + df["location"]
df["opacity"] = df["hazard_score_tsne"] / 100
df["opacity"] = df["opacity"].apply(lambda x: 0.6 if x < 0.6 else x)

event_total = df["event_date"].count()

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
                            [
                                html.P(
                                    "Select different locations via direct search or by selecting a specific oblast (administrative region) from the drop-down menu.",
                                    style={
                                        "color": "#DCDCDC",
                                        "marginBottom": "10px",
                                        "width": "114%",
                                        "fontSize": "18px",
                                    },
                                ),
                                html.P(
                                    "This map displays events across a date range specified by the date picker and you can filter out events via the hazard scale.",
                                    style={
                                        "color": "#DCDCDC",
                                        "marginBottom": "20px",
                                        "width": "110%",
                                        "fontSize": "18px",
                                    },
                                ),
                                dcc.Input(
                                    id="search-bar",
                                    type="text",
                                    debounce=True,
                                    placeholder="Enter a location",
                                    style={
                                        "width": "288px",
                                        "marginBottom": "20px",
                                        "paddingLeft": "8px",
                                        "paddingRight": "8px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="region-dropdown",
                                    options=[
                                        {"label": region, "value": region}
                                        for region in df["admin2"].unique()
                                    ],
                                    multi=True,
                                    placeholder="Select a region",
                                    style={
                                        "backgroundColor": "#3B3B3B",
                                        "color": "#757575",
                                        "width": "288px",
                                        "marginBottom": "16px",
                                    },
                                ),
                                dcc.DatePickerRange(
                                    id="date-picker",
                                    min_date_allowed=df["event_date"].min(),
                                    max_date_allowed=df["event_date"].max(),
                                    initial_visible_month=df["event_date"].min(),
                                    start_date=df["event_date"].min(),
                                    end_date=df["event_date"].max(),
                                    style={
                                        "marginBottom": "20px",
                                    },
                                ),
                                html.P(
                                    [
                                        html.Span(
                                            "Total Events: ",
                                            style={"font-weight": "bold"},
                                        ),
                                        html.Span(
                                            str(event_total),
                                            style={
                                                "color": "#DCDCDC",
                                                "marginBottom": "20px",
                                                "fontSize": "18px",
                                            },
                                        ),
                                    ]
                                ),
                                html.P(
                                    [
                                        html.Span(
                                            "Displayed Events: ",
                                            style={"font-weight": "bold"},
                                        ),
                                        html.Span(
                                            id="selected-events",
                                            style={
                                                "color": "#DCDCDC",
                                                "marginBottom": "20px",
                                                "fontSize": "18px",
                                            },
                                        ),
                                    ]
                                ),
                                html.P(
                                    [
                                        html.Span(
                                            "Hazard Score Range: ",
                                            style={"font-weight": "bold"},
                                        ),
                                        html.Span(
                                            id="hazard-range",
                                            style={
                                                "color": "#DCDCDC",
                                                "marginBottom": "20px",
                                                "fontSize": "18px",
                                            },
                                        ),
                                    ]
                                ),
                                html.P(
                                    [
                                        html.Span(
                                            "Note:", style={"font-weight": "bold"}
                                        ),
                                        " The map displays hazard level of a location based on most recent events, meaning that if a high intensity "
                                        + "event occurred causing high hazard then such will be the hazard score for that location. This does not necessarily "
                                        + "mean that such is the overall hazard level in that location. In order to view "
                                        + "all hazard score changes associated with a location, click on a map marker and the table below will list "
                                        + "all of the events that have happened at that location during the time period when the data was collected.",
                                    ],
                                    style={
                                        "color": "#DCDCDC",
                                        "fontSize": "18px",
                                    },
                                ),
                                html.P(
                                    [
                                        " In order to view "
                                        + "all hazard score changes associated with a location, click on a map marker and the table below will list "
                                        + "all of the events that have happened at that location during the time period when the data was collected.",
                                    ],
                                    style={
                                        "color": "#DCDCDC",
                                        "marginBottom": "20px",
                                        "fontSize": "18px",
                                    },
                                ),
                                html.P(
                                    [
                                        html.Span(
                                            "Clarification on the appearance of Crimea on the map:",
                                            style={"font-weight": "bold"},
                                        ),
                                        " According to Mapbox, Crimea is a contested territory and therefore it is automatically displayed as neither belonging to Ukraine or Russia.",
                                    ],
                                    style={
                                        "color": "#DCDCDC",
                                        "fontSize": "18px",
                                    },
                                ),
                                html.P(
                                    [
                                        " My word? Crimea is Ukraine. ",
                                    ],
                                    style={
                                        "color": "#DCDCDC",
                                        "marginBottom": "20px",
                                        "fontSize": "18px",
                                    },
                                ),
                                html.P(
                                    [
                                        html.Span("Source of Data: "),
                                        html.A(
                                            "ACLED",
                                            href="https://www.acleddata.com/",
                                            target="_blank",
                                        ),
                                    ],
                                    style={
                                        "color": "#DCDCDC",
                                        "marginBottom": "20px",
                                        "fontSize": "18px",
                                    },
                                ),
                            ],
                            style={"max-width": "460px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flex-direction": "column",
                        "align-items": "right",
                        "padding-left": "50px",
                    },
                ),
                # width=3,
                # style={'padding-left': '160px'},
                # xs=3,
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.Div(
                                    dcc.Graph(id="scatter-map", responsive=True),
                                    style={
                                        "margin-bottom": "20px",
                                    },
                                ),
                                html.Div(
                                    dcc.RangeSlider(
                                        id="hazard-score-slider",
                                        min=df["hazard_score_tsne"].min(),
                                        max=df["hazard_score_tsne"].max(),
                                        value=[
                                            df["hazard_score_tsne"].min(),
                                            df["hazard_score_tsne"].max(),
                                        ],
                                        marks={
                                            str(i): str(i)
                                            for i in range(
                                                int(df["hazard_score_tsne"].min()),
                                                int(df["hazard_score_tsne"].max()) + 1,
                                                5,
                                            )
                                        },
                                        step=5,
                                    ),
                                    style={
                                        "margin-bottom": "20px",
                                    },
                                ),
                                dash_table.DataTable(
                                    id="table",
                                    columns=[
                                        {"name": i, "id": i}
                                        for i in [
                                            "location",
                                            "event_date",
                                            "hazard_score_tsne",
                                        ]
                                    ],
                                    data=[],
                                    page_size=5,
                                    style_data_conditional=[
                                        {
                                            "if": {"row_index": "odd"},
                                            "backgroundColor": "rgb(50, 50, 50)",
                                        }
                                    ],
                                    style_header={
                                        "backgroundColor": "rgb(30, 30, 30)",
                                        "color": "white",
                                    },
                                    style_cell={
                                        "backgroundColor": "rgb(50, 50, 50)",
                                        "color": "white",
                                    },
                                    style_table={
                                        "width": "100%",
                                        "margin-bottom": "20px",
                                        "padding-left": "2px",
                                    },
                                ),
                                dbc.Button(
                                    "Clear",
                                    id="close-button",
                                    color="secondary",
                                    className="mr-1",
                                    style={"margin-bottom": "20px"},
                                ),
                            ],
                            style={"overflow": "auto"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flex-direction": "column",
                        "align-items": "left",
                        "padding-left": "20px",
                        "padding-right": "30px",
                    },
                    xl=6,
                    md=12,
                    lg=12,
                    sm=12,
                    xs=12,
                ),
            ],
            style={"padding-top": "40px"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P(
                            "Methodology",
                            style={
                                "display": "flex",
                                "justify-content": "center",
                                "align-items": "center",
                                "color": "#DCDCDC",
                                "marginBottom": "10px",
                                "fontSize": "34px",
                                "font-weight": "bold",
                            },
                        ),
                    ]
                )
            ],
            style={"padding-top": "80px"},
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Markdown(
                            """
                    &nbsp;&nbsp;&nbsp;&nbsp;The generation of the hazard_score_tsne variable was a process that involved several stages of data collection, preprocessing, feature engineering, and dimensional reduction.
                    My goal was to create a comprehensive hazard score that could reflect the danger level in different regions of Ukraine, based on geolocated conflict data.
                """,
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            [
                                "The process began with data collection. I gathered data on conflict-related events in Ukraine, thanks to the Armed Conflict Location & Event Data Project (ACLED). ",
                                html.A(
                                    " ACLED",
                                    href="https://acleddata.com/",
                                    target="_blank",
                                ),
                                " is a highly respected organization providing real-time data and analysis on current political violence and protest across the world. I am grateful for their support"
                                " in making this data available to me free of charge. Their data on the location, number of casualties, number of shelling events were"
                                " invaluable for this analysis. These factors were carefully chosen because they have a direct impact on the danger level of a region.",
                            ],
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "The collected data was then preprocessed. Using Python libraries such as NumPy and pandas, I cleaned, processed,"
                            " and formatted the data into a suitable structure for further analysis. This included handling missing values, converting data types,"
                            " and standardizing the data.",
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "During the feature engineering phase, I created new features from existing data that could improve the performance of my model. For instance,"
                            " I used spaCy's Natural Language Processing (NLP) capabilities to extract wounded data from text."
                            " This allowed me to create a more nuanced and granular measure of conflict impact in each location.",
                            style={"marginBottom": "20px"},
                        ),
                    ],
                    style={
                        "justify-content": "column",
                        "align-items": "left",
                        "color": "#DCDCDC",
                        "fontSize": "18px",
                        "padding-left": "40px",
                        "width": "80%",
                    },
                ),
                dbc.Col(
                    [
                        html.P(
                            "I performed dimensionality reduction using t-Distributed Stochastic Neighbor Embedding (t-SNE),"
                            " a machine learning algorithm that reduces high-dimensional data into a two or three-dimensional space,"
                            " making it easier to visualize and interpret. This allowed me to synthesize my many features into a single,"
                            " comprehensive 'hazard_score_tsne'.",
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "Once the hazard score was generated, it underwent several evaluations to ensure its validity. Feedback from Ukrainians confirmed"
                            " that the generated hazard scores align well with their perceptions of danger levels in different regions, affirming the model's"
                            " face validity. Additionally, the model demonstrated historical validation by aligning well with"
                            " historical conflict patterns.",
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "The model also exhibited spatial consistency, as areas close to each other had similar hazard scores unless divided"
                            " by clear demarcation lines like boundaries. This consistency in hazard scores across geographical proximity further"
                            " affirmed the reliability of the model's outputs.",
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "The result was a hazard score that encapsulated the key elements of conflict impact in each region."
                            " This comprehensive score is now aiding civilians and humanitarian organizations in making informed"
                            " decisions, allocating resources, and planning strategically in war-torn zones.",
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "Author: Yegor Smertenko",
                            style={"textAlign": "right", "marginRight": "40px"},
                        ),
                    ],
                    style={
                        "justify-content": "column",
                        "align-items": "right",
                        "color": "#DCDCDC",
                        "fontSize": "18px",
                        "padding-left": "40px",
                        "width": "80%",
                    },
                ),
            ],
            style={"padding-top": "50px"},
        ),
    ],
    fluid=True,
)


mapbox_access_token = os.getenv("MAPBOX_ACCESS_TOKEN")
custom_mapbox_style_url = os.getenv("MAPBOX_STYLE")


@app.callback(
    Output("scatter-map", "figure"),
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
        (dff["hazard_score_tsne"] >= hazard_score[0])
        & (dff["hazard_score_tsne"] <= hazard_score[1])
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
        + dff["hazard_score_tsne"].astype(str)
        + "<br>Date: "
        + dff["event_date"].astype(str)
    )

    selected_events = str(len(dff))

    hazard_range = str(hazard_score[0]) + " - " + str(hazard_score[1])

    fig = go.Figure(
        data=go.Scattermapbox(
            lat=dff["latitude"],
            lon=dff["longitude"],
            mode="markers",
            marker=go.scattermapbox.Marker(
                size=7,  # Adjust size as needed
                color=dff["hazard_score_tsne"],
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

    return fig, selected_events, hazard_range


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
            # dff = dff.rename(columns={'event_date': 'Date', 'hazard_score_tsne': 'Hazard Score'})
            return dff.to_dict("records")

    # if callback was triggered by close-button n_clicks
    elif ctx.triggered[0]["prop_id"] == "close-button.n_clicks":
        return []  # return an empty list when the close button is clicked

    else:
        return []  # return an empty list when no marker is clicked


# Run the app
if __name__ == "__main__":
    app.run_server(debug=False)
