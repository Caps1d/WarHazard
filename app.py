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
commands_dir = os.path.join(dir_path, "hazard_scores", "Hazards_latest.csv")


# Load your data
df = pd.read_csv(commands_dir)
df["event_date"] = pd.to_datetime(df["event_date"])

df["unique_id"] = df["admin2"] + "-" + df["admin3"] + "-" + df["location"]
df["opacity"] = df["hazard_score"] / 100
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
                                        "fontSize": "18px",
                                    },
                                ),
                                html.P(
                                    "This map displays events across a date range specified by the date picker and you can filter out events via the hazard scale.",
                                    style={
                                        "color": "#DCDCDC",
                                        "marginBottom": "20px",
                                        "fontSize": "18px",
                                    },
                                ),
                                dcc.Input(
                                    id="search-bar",
                                    type="text",
                                    debounce=True,
                                    placeholder="Enter a location",
                                    style={
                                        "width": "100%",
                                        "marginBottom": "20px",
                                        "paddingLeft": "8px",
                                        "paddingRight": "8px",
                                        "maxWidth": "460px",
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
                                        "marginBottom": "16px",
                                        "maxWidth": "460px",
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
                                    style={
                                        "margin-bottom": "20px",
                                    },
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                html.Div(
                                    dcc.RangeSlider(
                                        id="hazard-score-slider",
                                        min=df["hazard_score"].min(),
                                        max=df["hazard_score"].max(),
                                        value=[
                                            df["hazard_score"].min(),
                                            df["hazard_score"].max(),
                                        ],
                                        marks={
                                            str(i): str(i)
                                            for i in range(
                                                0,
                                                100,
                                                10,
                                            )
                                        },
                                        step=10,
                                    ),
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
                                    dash_table.DataTable(
                                        id="table",
                                        columns=[
                                            {"name": i, "id": i}
                                            for i in [
                                                "location",
                                                "event_date",
                                                "hazard_score",
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
                                            "margin-bottom": "20px",
                                            "padding-left": "2px",
                                        },
                                    ),
                                    style={"margin-bottom": "20px"},
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Clear",
                                        id="close-button",
                                        color="secondary",
                                        className="mr-1",
                                        style={"margin-bottom": "20px"},
                                    ),
                                    width="auto",
                                ),
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
                        html.P(
                            "Project Overview",
                            style={
                                "text-align": "center",
                                "align-items": "center",
                                "color": "#DCDCDC",
                                "marginBottom": "10px",
                                "fontSize": "34px",
                                "font-weight": "bold",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.P(
                            "The aim of this project was to provide a hazard score for all locations in Ukraine that were involved in war-related events such battles, explosions/remote violence events, violence against civilians and factors that pose risk to civilians living in those locations. The idea is to take those factors and condense them into one readable and comprehensible score which would reflect the danger level of the associated location.",
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "The main issue I have faced while developing a statistical method that could provide such a score was to combine domain knowledge and account for the known unknowns. In complex events like wars, it's very difficult to obtain a comprehensive dataset that can capture the underlying dynamics. Not because of lack of data but due to the abundance and sheer number of factors involved that are either too hard or too expensive to account for. Furthermore, the issue of inherent bias involved in various sources is also prevalent and poses a serious risk of polluting a dataset. Hence, besides accounting for all factors that contribute to an event, one must also consider how valid the data that they’re using is. Therefore, I decided to use the data provided to me by ACLED for non-commercial purposes, the aim of this project is for data exploration, visualization and awareness. ACLED is a non-profit organization that has been collecting data on armed conflict events since 2014. For further information on ACLED please refer to the hyperlink above. ",
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "ACLED’s raw dataset was comprehensive enough to include three levels of administrative borders for each location, event types, sub events, dates, casualties, presence of civilian targeting, what actors were involved in the event, and a text based note column that included additional information pertaining to the event. As mentioned before, due the complexity and nature of armed conflicts there are a significant number of factors that cannot be accounted for and some are just not available to the public. Namely for this project, ideally, I would need access to air defense numbers in a location, overall equipment concentration, the number of missiles that have been launched in a remote violence event, where they were launched from, how many have been shot down, logistic routes and hubs and this is just to name a few. Some of this data is available to the public, but the sources are sparse and inconsistent. In other words, the known unknowns. Hence, for now I have decided to use the baseline data set from ACLED and process the data enough to account for some of those factors I mentioned earlier.",
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "In later updates to this project, I aim to include some new features such as whether a location is part of a logistic route, missile count and type and reduce the number of hardcoded points that are used to account for domain knowledge on the war in Ukraine. Furthermore, for people interested in numbers, I will also include various plots to show event distributions, frequencies and more visualizations of various relationships that are present in the dataset. Nevertheless, I am happy with how this project has turned out and its accuracy in reflecting the danger level of locations across Ukraine.",
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "Quick note on the project's implementation: everything was done using python and various data processing libraries, numpy and pandas being at the core of it and plotly’s dash was used for making the web app.",
                        ),
                    ],
                    style={
                        "text-align": "left",
                        "align-items": "left",
                        "color": "#DCDCDC",
                        "padding-left": "40px",
                        "padding-right": "40px",
                        "fontSize": "18px",
                    },
                ),
            ],
            style={
                "justify-content": "center",
                "padding-top": "80px",
            },
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
                ),
            ],
            style={
                "padding-top": "40px",
                "padding-bottom": "6px",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Markdown(
                            """
                            First, in order to obtain the raw dataset I had to write a python script that uses ACLED’s API and a list of query filters that will filter out any data that I find irrelevant for this project or the dataset batch I am working on. I made different batches of data since the events in Ukraine would also develop in separate phases. For example, UAF’s (Ukrainian Armed Forces) summer offensive in 2022 which lead to the reclamation of a significant portion of the Kharkiv region presents a different state of events than what happened during the winter of 2023 with Russia’s winter offensive and the battle for Bakmut involving Wagner pmc’s. Furthermore, some data like iso (country code of the event), event id, and a few other id columns were filtered out from the dataset due to their irrelevance for the task at hand alongside with source type (source of the event report), source scale, and inter 1 and inter2. The reason for dropping some of the non-id related columns was due to them not contributing to the actual relationships between the data I was exploring and some features that were included already carry the context of those dropped features.
                """,
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "The obtained filtered dataset includes the following baseline features (i.e columns): event date, event type (Battles, Explosions/Remote violence, Violence against civilians), sub event type (armed clashes, shelling/artillery/missile attack, etc.), actor1, actor2, admin1, admin2, admin3 (the 3 admin columns are representing 3 levels of administrative boundaries of the location), location, latitude, longitude, notes (short description of the event), fatalities, civilian targeting (whether civilians were the primary target or not).",
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "As my first step in data preprocessing I have read the json dataset and converted it into a dataframe using pandas (think of it like an excel sheet). I have then explored the datatypes of all columns and converted fatalities, latitude and longitude to numeric type. Then after exploring the dataset, I noticed that the notes column would sometimes include data on casualties. This was an important finding since I knew that some events actually had a significant number of civilian casualties that weren’t counting as fatalities and overall a significant number was never reported either, due to the lack of clarity in numbers varying between sources. Hence, even when some severe events like missile strikes or battles have occurred that had large numbers of casualties reported by external sources, the raw dataset would have 0 fatalities. In order to account best for casualties and affected parties, I have opted-in to try and extract this data from the text-based notes column. Due to varying formatting and wording something like regular expressions could not be used as a lot of the entries would get ignored due to formatting. Therefore, I have used SpaCy’s NLP machine learning model to do the trick. After setting the proper conditions for the token, the model was able to extract wounded data from the notes column and thus I created a new feature called ‘wounded’ which represents the number of wounded individuals involved in the event.",
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "Next step was to aggregate the data. As mentioned before, each row would have an event date and thus I thought it’s best to aggregate the data by month for a location to condense the data a bit. This led to the creation of 2 new aggregate features: number of events and total casualties.",
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "As mentioned before, there was a certain amount of domain knowledge that needed to be incorporated into the statistical approach designed for this project. Hence, I decided to create a coordinate list of locations that are well-known hotzones, such as Bakhmut, Kupiansk, Avdiivka, Vuhledar, Robotyne, and Soledar. The reasoning behind this is to use it as a reference point for a location's proximity to the battlefront. Those locations and a lot of others that neighbor them are known engagement zones which would have a hazard score of 100. In order to obtain the distance between a location and a hot zone I have used the haversine function which measures the distance between two sets of coordinates while accounting for earth’s curvature. After obtaining the distances of a location to each of the hot zones, I have selected the shortest distance out of all of them and created a new feature called minimal distance to the battlefront.",
                            style={"marginBottom": "20px"},
                        ),
                        html.P(
                            "Now that new features have been added, the next step was to encode some categorical features like event type and civilian targeting into numerical values using one-hot encoding. This way it's easier for the model to read the features as it can check whether a specific type of event has occurred simply by setting 1 or 0 in the respective column. Furthermore, I combined explosions/remote violence events with civilian targeting to differentiate between missile strikes on civilian infrastructure and shelling events at the frontline. The idea is that if civilian targeting has a true value in the same row where the explosion event is also registered then that means it's most likely a remote violence event caused by missile strikes on civilian infrastructure away from the frontline. The logic behind this is that it will be very unlikely that you will have a registered explosion event involving civilians at the front lines and thus any explosion/remote violence event that has false value for the civilian targeting column most likely points to an artillery event at the front lines, in other words, a high hazard zone. This was later confirmed by artillery sub event type in rows that have that event type and false for civilian targeting.",
                            style={"marginBottom": "20px"},
                        ),
                    ],
                    style={
                        "justify-content": "column",
                        "align-items": "left",
                        "color": "#DCDCDC",
                        "fontSize": "18px",
                    },
                ),
                html.Div(
                    [
                        html.P(
                            "I performed dimensionality reduction using t-Distributed Stochastic Neighbor Embedding (t-SNE),"
                            " a machine learning algorithm that reduces high-dimensional data into a two or three-dimensional space,"
                            " making it easier to visualize and interpret. This allowed me to synthesize my many features into a single,"
                            " comprehensive 'hazard_score'.",
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
                    },
                ),
            ],
            style={
                # "padding-top": "20px",
                "padding-left": "40px",
                "padding-right": "40px",
            },
            className="methodology",
        ),
        html.Script(
            """
        window.onload = function() {
            window.dispatchEvent(new Event('resize'));
        };
    """
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
            return dff.to_dict("records")

    # if callback was triggered by close-button n_clicks
    elif ctx.triggered[0]["prop_id"] == "close-button.n_clicks":
        return []  # return an empty list when the close button is clicked

    else:
        return []  # return an empty list when no marker is clicked


# Run the app
if __name__ == "__main__":
    app.run_server(debug=False)
