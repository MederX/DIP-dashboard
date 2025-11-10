import json
import time
import uuid
from copy import deepcopy

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

import dash_reusable_components as drc
from utils import STORAGE_PLACEHOLDER, GRAPH_PLACEHOLDER, IMAGE_STRING_PLACEHOLDER
from utils import (apply_filters, show_histogram, generate_lasso_mask, apply_enhancements, 
                  apply_resolution_scaling, apply_log_transformation, apply_histogram_equalization,
                  apply_histogram_matching, add_noise, reduce_noise)

DEBUG = True

app = dash.Dash(__name__)
server = app.server

# Simple in-memory storage for images (per session)
local_image_storage = {}

def serve_layout():
    # Generates a session ID
    session_id = str(uuid.uuid4())
    
    # Store the default image for this session
    local_image_storage[session_id] = IMAGE_STRING_PLACEHOLDER
    
    if DEBUG:
        print(f"Created new session: {session_id}")

    # App Layout
    return html.Div([
        # Session ID
        html.Div(session_id, id='session-id', style={'display': 'none'}),

        # Banner display
        html.Div([
            html.H2(
                'Image Processing App',
                id='title'
            ),
            html.Div(
                "🖼️ Local Image Processing Tool",
                style={'color': '#7FDBFF', 'fontSize': '18px'}
            )
        ],
            className="banner"
        ),

        # Body
        html.Div(className="container", children=[
            html.Div(className='row', children=[
                html.Div(className='five columns', children=[
                    drc.Card([
                        dcc.Upload(
                            id='upload-image',
                            children=[
                                'Drag and Drop or ',
                                html.A('Select an Image')
                            ],
                            style={
                                'width': '100%',
                                'height': '50px',
                                'lineHeight': '50px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center'
                            },
                            accept='image/*'
                        ),

                        drc.NamedInlineRadioItems(
                            name='Selection Mode',
                            short='selection-mode',
                            options=[
                                {'label': ' Rectangular', 'value': 'select'},
                                {'label': ' Lasso', 'value': 'lasso'}
                            ],
                            val='select'
                        ),

                        drc.NamedInlineRadioItems(
                            name='Image Display Format',
                            short='encoding-format',
                            options=[
                                {'label': ' JPEG', 'value': 'jpeg'},
                                {'label': ' PNG', 'value': 'png'}
                            ],
                            val='jpeg'
                        ),
                    ]),

                    drc.Card([
                        drc.CustomDropdown(
                            id='dropdown-filters',
                            options=[
                                {'label': 'Blur', 'value': 'blur'},
                                {'label': 'Contour', 'value': 'contour'},
                                {'label': 'Detail', 'value': 'detail'},
                                {'label': 'Enhance Edge', 'value': 'edge_enhance'},
                                {'label': 'Enhance Edge (More)', 'value': 'edge_enhance_more'},
                                {'label': 'Emboss', 'value': 'emboss'},
                                {'label': 'Find Edges', 'value': 'find_edges'},
                                {'label': 'Sharpen', 'value': 'sharpen'},
                                {'label': 'Smooth', 'value': 'smooth'},
                                {'label': 'Smooth (More)', 'value': 'smooth_more'}
                            ],
                            searchable=False,
                            placeholder='Basic Filter...'
                        ),

                        drc.CustomDropdown(
                            id='dropdown-enhance',
                            options=[
                                {'label': 'Brightness', 'value': 'brightness'},
                                {'label': 'Color Balance', 'value': 'color'},
                                {'label': 'Contrast', 'value': 'contrast'},
                                {'label': 'Sharpness', 'value': 'sharpness'}
                            ],
                            searchable=False,
                            placeholder='Enhance...'
                        ),

                        html.Div(
                            id='div-enhancement-factor',
                            style={
                                'display': 'none',
                                'margin': '25px 5px 30px 0px'
                            },
                            children=[
                                f"Enhancement Factor:",
                                html.Div(
                                    style={'margin-left': '5px'},
                                    children=dcc.Slider(
                                        id='slider-enhancement-factor',
                                        min=0,
                                        max=2,
                                        step=0.1,
                                        value=1,
                                        updatemode='drag'
                                    )
                                )
                            ]
                        ),

                        html.Div(
                            id='div-spatial-resolution',
                            style={'margin': '25px 5px 30px 0px'},
                            children=[
                                f"Spatial Resolution (%):",
                                html.Div(
                                    style={'margin-left': '5px'},
                                    children=dcc.Slider(
                                        id='slider-spatial-resolution',
                                        min=10,
                                        max=100,
                                        step=5,
                                        value=100,
                                        updatemode='drag',
                                        marks={
                                            10: '10%',
                                            25: '25%', 
                                            50: '50%',
                                            75: '75%',
                                            100: '100%'
                                        },
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                )
                            ]
                        ),

                        drc.NamedInlineRadioItems(
                            name='Log Transformation',
                            short='log-transformation',
                            options=[
                                {'label': ' Off', 'value': 'off'},
                                {'label': ' Standard', 'value': 'standard'},
                                {'label': ' Enhanced', 'value': 'enhanced'}
                            ],
                            val='off',
                            style={'margin': '15px 0px'}
                        ),

                        html.Div(
                            id='div-log-constant',
                            style={
                                'display': 'none',
                                'margin': '10px 5px 20px 0px'
                            },
                            children=[
                                f"Log Constant (c):",
                                html.Div(
                                    style={'margin-left': '5px'},
                                    children=dcc.Slider(
                                        id='slider-log-constant',
                                        min=0.5,
                                        max=3.0,
                                        step=0.1,
                                        value=1.0,
                                        updatemode='drag',
                                        marks={
                                            0.5: '0.5',
                                            1.0: '1.0',
                                            1.5: '1.5',
                                            2.0: '2.0',
                                            2.5: '2.5',
                                            3.0: '3.0'
                                        },
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                )
                            ]
                        ),

                        # Enhancement Operations Section
                        html.Hr(),
                        html.H5('Enhancement Operations'),
                        
                        drc.NamedInlineRadioItems(
                            name='Histogram Equalization',
                            short='histogram-equalization',
                            options=[
                                {'label': ' Off', 'value': 'off'},
                                {'label': ' On', 'value': 'on'}
                            ],
                            val='off',
                            style={'margin': '10px 0px'}
                        ),

                        drc.NamedInlineRadioItems(
                            name='Histogram Matching',
                            short='histogram-matching',
                            options=[
                                {'label': ' Off', 'value': 'off'},
                                {'label': ' On', 'value': 'on'}
                            ],
                            val='off',
                            style={'margin': '10px 0px'}
                        ),
                        
                        dcc.Upload(
                            id='upload-reference-image',
                            children=[
                                'Drag and Drop or ',
                                html.A('Select a Reference Image')
                            ],
                            style={
                                'width': '100%',
                                'height': '50px',
                                'lineHeight': '50px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin-bottom': '10px'
                            },
                            accept='image/*'
                        ),

                        # Image Restoration Section
                        html.Hr(),
                        html.H5('Image Restoration'),

                        html.Div([
                            html.Label('Add Noise:', style={'margin-bottom': '5px'}),
                            drc.CustomDropdown(
                                id='dropdown-add-noise',
                                options=[
                                    {'label': 'None', 'value': ''},
                                    {'label': 'Gaussian', 'value': 'gaussian'},
                                    {'label': 'Salt & Pepper', 'value': 'salt_pepper'},
                                    {'label': 'Poisson', 'value': 'poisson'},
                                    {'label': 'Uniform', 'value': 'uniform'}
                                ],
                                searchable=False,
                                placeholder="Select noise type..."
                            )
                        ], style={'margin': '10px 0px'}),

                        html.Div(
                            id='div-noise-intensity',
                            style={
                                'display': 'none',
                                'margin': '10px 5px 20px 0px'
                            },
                            children=[
                                f"Noise Intensity:",
                                html.Div(
                                    style={'margin-left': '5px'},
                                    children=dcc.Slider(
                                        id='slider-noise-intensity',
                                        min=1,
                                        max=50,
                                        step=1,
                                        value=15,
                                        updatemode='drag',
                                        marks={
                                            1: '1',
                                            10: '10',
                                            20: '20',
                                            30: '30',
                                            40: '40',
                                            50: '50'
                                        },
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                )
                            ]
                        ),

                        html.Div([
                            html.Label('Noise Reduction:', style={'margin-bottom': '5px'}),
                            drc.CustomDropdown(
                                id='dropdown-noise-reduction',
                                options=[
                                    {'label': 'None', 'value': ''},
                                    {'label': 'Gaussian', 'value': 'gaussian'},
                                    {'label': 'Median', 'value': 'median'},
                                    {'label': 'Bilateral', 'value': 'bilateral'},
                                    {'label': 'Wiener', 'value': 'wiener'}
                                ],
                                searchable=False,
                                placeholder="Select noise reduction..."
                            )
                        ], style={'margin': '10px 0px'}),

                        html.Div(
                            id='div-noise-reduction-strength',
                            style={
                                'display': 'none',
                                'margin': '10px 5px 20px 0px'
                            },
                            children=[
                                f"Reduction Strength:",
                                html.Div(
                                    style={'margin-left': '5px'},
                                    children=dcc.Slider(
                                        id='slider-noise-reduction-strength',
                                        min=0.1,
                                        max=3.0,
                                        step=0.1,
                                        value=1.0,
                                        updatemode='drag',
                                        marks={
                                            0.1: '0.1',
                                            0.5: '0.5',
                                            1.0: '1.0',
                                            1.5: '1.5',
                                            2.0: '2.0',
                                            3.0: '3.0'
                                        },
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    )
                                )
                            ]
                        ),

                        html.Button(
                            'Run Operation',
                            id='button-run-operation',
                            style={'margin-right': '10px', 'margin-top': '5px'}
                        ),

                        html.Button(
                            'Undo',
                            id='button-undo',
                            style={'margin-top': '5px'}
                        )
                    ]),
                ]),

                html.Div(
                    className='seven columns',
                    style={'float': 'right'},
                    children=[
                        # The Interactive Image Div contains the dcc Graph
                        # showing the image, as well as the hidden div storing
                        # the true image
                        html.Div(id='div-interactive-image', children=[
                            GRAPH_PLACEHOLDER,
                            html.Div(
                                id='div-storage',
                                children=STORAGE_PLACEHOLDER,
                                style={'display': 'none'}
                            )
                        ]),
                        dcc.Graph(id='graph-histogram-colors',
                              config={'displayModeBar': False})
                    ]
                )
            ])
        ])
    ])


app.layout = serve_layout


# Helper functions for callbacks
def add_action_to_stack(action_stack, operation, operation_type, selectedData):
    """
    Add new action to the action stack, in-place.
    """
    new_action = {
        'operation': operation,
        'type': operation_type,
        'selectedData': selectedData
    }
    action_stack.append(new_action)


def undo_last_action(n_clicks, storage):
    action_stack = storage['action_stack']

    if n_clicks is None:
        storage['undo_click_count'] = 0

    # If the stack isn't empty and the undo click count has changed
    elif len(action_stack) > 0 and n_clicks > storage['undo_click_count']:
        # Remove the last action on the stack
        action_stack.pop()
        # Update the undo click count
        storage['undo_click_count'] = n_clicks

    return storage


def apply_actions_on_image(session_id, action_stack, filename, image_signature):
    """
    Recursively apply actions to the image stored in local memory.
    """
    action_stack = deepcopy(action_stack)

    # If we have arrived to the original image
    if len(action_stack) == 0:
        # Get the image from local storage
        if session_id in local_image_storage:
            im_pil = drc.b64_to_pil(local_image_storage[session_id])
            return im_pil
        else:
            # Fallback to default image
            im_pil = drc.b64_to_pil(IMAGE_STRING_PLACEHOLDER)
            return im_pil

    # Pop out the last action
    last_action = action_stack.pop()
    
    # Apply all the previous actions recursively
    im_pil = apply_actions_on_image(session_id, action_stack, filename, image_signature)
    im_size = im_pil.size

    # Apply the current action
    operation = last_action['operation']
    selectedData = last_action['selectedData']
    action_type = last_action['type']

    # Determine selection zone
    # Select using Lasso
    if selectedData and 'lassoPoints' in selectedData:
        selection_mode = 'lasso'
        selection_zone = generate_lasso_mask(im_pil, selectedData)
    # Select using rectangular box
    elif selectedData and 'range' in selectedData:
        selection_mode = 'select'
        lower, upper = map(int, selectedData['range']['y'])
        left, right = map(int, selectedData['range']['x'])
        # Adjust height difference
        height = im_size[1]
        upper = height - upper
        lower = height - lower
        selection_zone = (left, upper, right, lower)
    # Select the whole image
    else:
        selection_mode = 'select'
        selection_zone = (0, 0) + im_size

    # Apply the operations
    if action_type == 'filter':
        apply_filters(
            image=im_pil,
            zone=selection_zone,
            filter=operation,
            mode=selection_mode
        )
    elif action_type == 'enhance':
        enhancement = operation['enhancement']
        factor = operation['enhancement_factor']

        apply_enhancements(
            image=im_pil,
            zone=selection_zone,
            enhancement=enhancement,
            enhancement_factor=factor,
            mode=selection_mode
        )
    elif action_type == 'resolution':
        resolution_factor = operation['resolution_factor']
        # Apply resolution scaling to the entire image
        im_pil = apply_resolution_scaling(im_pil, resolution_factor)
        return im_pil

    return im_pil


@app.callback(Output('interactive-image', 'figure'),
              [Input('radio-selection-mode', 'value')],
              [State('interactive-image', 'figure')])
def update_selection_mode(selection_mode, figure):
    if figure:
        figure['layout']['dragmode'] = selection_mode
    return figure


@app.callback(Output('graph-histogram-colors', 'figure'),
              [Input('interactive-image', 'figure')])
def update_histogram(figure):
    # Check if figure exists and has the expected structure
    if not figure or 'layout' not in figure or 'images' not in figure['layout'] or not figure['layout']['images']:
        # Return a default histogram for the default image
        im_pil = drc.b64_to_pil(IMAGE_STRING_PLACEHOLDER)
        return show_histogram(im_pil)
    
    try:
        # Retrieve the image stored inside the figure
        enc_str = figure['layout']['images'][0]['source'].split(';base64,')[-1]
        # Creates the PIL Image object from the b64 png encoding
        im_pil = drc.b64_to_pil(string=enc_str)
        return show_histogram(im_pil)
    except (IndexError, KeyError, AttributeError):
        # Fallback to default image if there's an error
        im_pil = drc.b64_to_pil(IMAGE_STRING_PLACEHOLDER)
        return show_histogram(im_pil)


@app.callback(Output('div-interactive-image', 'children'),
              [Input('upload-image', 'contents'),
               Input('button-undo', 'n_clicks'),
               Input('button-run-operation', 'n_clicks'),
               Input('slider-spatial-resolution', 'value'),
               Input('radio-log-transformation', 'value'),
               Input('slider-log-constant', 'value'),
               Input('radio-histogram-equalization', 'value'),
               Input('radio-histogram-matching', 'value'),
               Input('dropdown-add-noise', 'value'),
               Input('slider-noise-intensity', 'value'),
               Input('dropdown-noise-reduction', 'value'),
               Input('slider-noise-reduction-strength', 'value')],
              [State('interactive-image', 'selectedData'),
               State('dropdown-filters', 'value'),
               State('dropdown-enhance', 'value'),
               State('slider-enhancement-factor', 'value'),
               State('upload-image', 'filename'),
               State('upload-reference-image', 'contents'),
               State('radio-selection-mode', 'value'),
               State('radio-encoding-format', 'value'),
               State('div-storage', 'children'),
               State('session-id', 'children')])
def update_graph_interactive_image(content, undo_clicks, n_clicks, spatial_resolution, 
                                   log_transformation, log_constant, hist_equalization,
                                   hist_matching, add_noise_type, noise_intensity,
                                   noise_reduction_type, noise_reduction_strength, selectedData,
                                   filters, enhance, enhancement_factor, new_filename,
                                   reference_image_content, dragmode, enc_format, storage, session_id):
    t_start = time.time()

    # Retrieve information saved in storage
    storage = json.loads(storage)
    filename = storage['filename']
    image_signature = storage['image_signature']

    # Run undo function if needed
    storage = undo_last_action(undo_clicks, storage)

    # If a new file was uploaded
    if new_filename and new_filename != filename:
        if DEBUG:
            print(f"{filename} replaced by {new_filename}")

        # Update the storage dict
        storage['filename'] = new_filename

        # Parse the string and convert to PIL
        string = content.split(';base64,')[-1]
        im_pil = drc.b64_to_pil(string)

        # Update the image signature
        storage['image_signature'] = string[:200]

        # Store the image string locally
        local_image_storage[session_id] = string
        if DEBUG:
            print(f"{new_filename} stored locally for session {session_id}")

        # Reset the action stack
        storage['action_stack'] = []
        
        # Apply spatial resolution scaling to newly uploaded image if needed
        if spatial_resolution and spatial_resolution != 100:
            resolution_factor = spatial_resolution / 100.0
            im_pil = apply_resolution_scaling(im_pil, resolution_factor)
            
        # Apply log transformation to newly uploaded image if needed
        if log_transformation and log_transformation != 'off':
            if log_transformation == 'standard':
                c_value = 1.0
            elif log_transformation == 'enhanced':
                c_value = log_constant if log_constant else 1.5
            im_pil = apply_log_transformation(im_pil, c_value)
            
        # Apply histogram equalization if enabled
        if hist_equalization and hist_equalization == 'on':
            im_pil = apply_histogram_equalization(im_pil)
            
        # Apply histogram matching if enabled
        if hist_matching == 'on' and reference_image_content:
            ref_string = reference_image_content.split(';base64,')[-1]
            ref_im_pil = drc.b64_to_pil(ref_string)
            im_pil = apply_histogram_matching(im_pil, ref_im_pil)
            
        # Add noise if selected
        if add_noise_type:
            noise_int = noise_intensity if noise_intensity else 15
            im_pil = add_noise(im_pil, add_noise_type, noise_int)
            
        # Apply noise reduction if selected
        if noise_reduction_type:
            reduction_str = noise_reduction_strength if noise_reduction_strength else 1.0
            im_pil = reduce_noise(im_pil, noise_reduction_type, reduction_str)

    # If an operation was applied
    else:
        # Add actions to the action stack
        if filters:
            add_action_to_stack(storage['action_stack'], filters, 'filter', selectedData)

        if enhance:
            operation = {
                'enhancement': enhance,
                'enhancement_factor': enhancement_factor,
            }
            add_action_to_stack(storage['action_stack'], operation, 'enhance', selectedData)

        # Apply the required actions to the picture
        im_pil = apply_actions_on_image(session_id, storage['action_stack'], filename, image_signature)
        
        # Apply spatial resolution scaling if different from 100%
        if spatial_resolution and spatial_resolution != 100:
            resolution_factor = spatial_resolution / 100.0
            im_pil = apply_resolution_scaling(im_pil, resolution_factor)
            
        # Apply log transformation if enabled
        if log_transformation and log_transformation != 'off':
            if log_transformation == 'standard':
                c_value = 1.0
            elif log_transformation == 'enhanced':
                c_value = log_constant if log_constant else 1.5
            im_pil = apply_log_transformation(im_pil, c_value)
            
        # Apply histogram equalization if enabled
        if hist_equalization and hist_equalization == 'on':
            im_pil = apply_histogram_equalization(im_pil)
            
        # Apply histogram matching if enabled
        if hist_matching == 'on' and reference_image_content:
            ref_string = reference_image_content.split(';base64,')[-1]
            ref_im_pil = drc.b64_to_pil(ref_string)
            im_pil = apply_histogram_matching(im_pil, ref_im_pil)
            
        # Add noise if selected
        if add_noise_type:
            noise_int = noise_intensity if noise_intensity else 15
            im_pil = add_noise(im_pil, add_noise_type, noise_int)
            
        # Apply noise reduction if selected
        if noise_reduction_type:
            reduction_str = noise_reduction_strength if noise_reduction_strength else 1.0
            im_pil = reduce_noise(im_pil, noise_reduction_type, reduction_str)

    t_end = time.time()
    if DEBUG:
        print(f"Updated Image Storage in {t_end - t_start:.3f} sec")

    return [
        drc.InteractiveImagePIL(
            image_id='interactive-image',
            image=im_pil,
            enc_format=enc_format,
            display_mode='fixed',
            dragmode=dragmode,
            verbose=DEBUG
        ),
        html.Div(
            id='div-storage',
            children=json.dumps(storage),
            style={'display': 'none'}
        )
    ]


# Show/Hide Enhancement Factor Slider
@app.callback(Output('div-enhancement-factor', 'style'),
              [Input('dropdown-enhance', 'value')],
              [State('div-enhancement-factor', 'style')])
def show_slider_enhancement_factor(value, style):
    if value:
        style['display'] = 'block'
    else:
        style['display'] = 'none'
    return style


# Show/Hide Log Constant Slider
@app.callback(Output('div-log-constant', 'style'),
              [Input('radio-log-transformation', 'value')],
              [State('div-log-constant', 'style')])
def show_slider_log_constant(value, style):
    if value in ['standard', 'enhanced']:
        style['display'] = 'block'
    else:
        style['display'] = 'none'
    return style


# Show/Hide Noise Intensity Slider
@app.callback(Output('div-noise-intensity', 'style'),
              [Input('dropdown-add-noise', 'value')],
              [State('div-noise-intensity', 'style')])
def show_slider_noise_intensity(value, style):
    if value:
        style['display'] = 'block'
    else:
        style['display'] = 'none'
    return style


# Show/Hide Noise Reduction Strength Slider
@app.callback(Output('div-noise-reduction-strength', 'style'),
              [Input('dropdown-noise-reduction', 'value')],
              [State('div-noise-reduction-strength', 'style')])
def show_slider_noise_reduction_strength(value, style):
    if value:
        style['display'] = 'block'
    else:
        style['display'] = 'none'
    return style


# Reset Dropdown Callbacks
@app.callback(Output('dropdown-filters', 'value'),
              [Input('button-run-operation', 'n_clicks')])
def reset_dropdown_filters(_):
    return None


@app.callback(Output('dropdown-enhance', 'value'),
              [Input('button-run-operation', 'n_clicks')])
def reset_dropdown_enhance(_):
    return None


# Running the server
if __name__ == '__main__':
    app.run(debug=True)
