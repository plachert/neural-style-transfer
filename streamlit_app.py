"""Streamlit app for experimenting with NST algorithm."""
from __future__ import annotations

import pathlib

import streamlit as st
from activation_tracker.activation import SUPPORTED_FILTERS
from activation_tracker.model import ModelWithActivations

import nst.image_processing as img_proc
from nst.activation_filters import ConvLikeActivationFilter
from nst.activation_filters import VerboseActivationFilter
from nst.configs import SUPPORTED_CONFIGS
from nst.optimization import optimize_image


UPLOADED_PATH = pathlib.Path('examples/uploaded')
UPLOADED_PATH.mkdir(parents=True, exist_ok=True)


def run_nst(
    input_init: str,
    content_image_path: pathlib.Path,
    content_weight: float,
    style_image_path: pathlib.Path,
    style_weight: float,
    config_name: str,
    selected_content_layer: list,
    selected_style_layers: list,
    n_iterations: int,
    regularization_coeff: float,
    lr: float,
):
    content_filter = {
        'content': [
            ConvLikeActivationFilter(
            ), VerboseActivationFilter(selected_content_layer),
        ],
    }
    style_filter = {
        'style': [
            ConvLikeActivationFilter(
            ), VerboseActivationFilter(selected_style_layers),
        ],
    }
    activation_filters = content_filter | style_filter
    config = SUPPORTED_CONFIGS[config_name]
    classifier = config.classifier
    processor = config.processor
    deprocessor = config.deprocessor
    model_with_activations = ModelWithActivations(
        model=classifier, activation_filters=activation_filters,
    )
    content_image = processor(img_proc.load_image_from(content_image_path))
    _, h, w = content_image.shape
    style_image = processor(img_proc.load_image_from(style_image_path))
    style_image = img_proc.resize_to_image(content_image, style_image)
    if input_init == 'random image':
        input_image = img_proc.create_random_image(h, w)
    elif input_init == 'content image':
        input_image = content_image
    else:
        input_image = style_image
    images = optimize_image(
        content_image=content_image,
        content_weight=content_weight,
        style_image=style_image,
        style_weight=style_weight,
        input_image=input_image,
        model=model_with_activations,
        n_iterations=n_iterations,
        regularization_coeff=regularization_coeff,
        lr=lr,
    )
    images = [
        img_proc.channel_last(img_proc.convert_to_255scale(deprocessor(image)))
        for image in images
    ]
    return images


def run():
    with st.spinner('Processing...'):
        images = run_nst(
            input_init=input_init,
            content_image_path=content_image_path,
            content_weight=content_weight,
            style_image_path=style_image_path,
            style_weight=style_weight,
            config_name=model_selection,
            selected_content_layer=[selected_content_layer],
            selected_style_layers=selected_style_layers,
            n_iterations=n_iterations,
            regularization_coeff=regularization_coeff,
            lr=lr,
        )
        st.session_state['images'] = images
        st.session_state['last_run_params'] = {
            'Content image path': content_image_path,
            'Content weight': content_weight,
            'Style image path': style_image_path,
            'Style weight': style_weight,
            'Model': model_selection,
            'Content layer': selected_content_layer,
            'Style layers': selected_style_layers,
            'Number of iterations': n_iterations,
            'Regularization coeff': regularization_coeff,
            'Learning rate': lr,
        }


@st.cache_data
def get_strategy_params(config_name):
    config = SUPPORTED_CONFIGS[config_name]
    model = config.classifier
    example_input = config.example_input
    activation_filter_class = VerboseActivationFilter
    activation_filters = {'convlike': [ConvLikeActivationFilter()]}
    model_with_activations = ModelWithActivations(
        model=model,
        activation_filters=activation_filters,
        example_input=example_input,
    )
    activations = model_with_activations.activations['convlike']
    parameters = activation_filter_class.list_all_available_parameters(
        activations,
    )
    return parameters


def get_path_from_uploaded(uploaded_file):
    if uploaded_file is not None:
        with open(f'{UPLOADED_PATH}/{uploaded_file.name}', 'wb') as f:
            f.write(uploaded_file.read())
        image_path = f'{UPLOADED_PATH}/{uploaded_file.name}'
    else:
        image_path = None
    return image_path


if __name__ == '__main__':
    uploaded_path = pathlib.Path('examples/uploaded')
    uploaded_path.mkdir(parents=True, exist_ok=True)
    st.set_page_config(
        layout='wide',
        initial_sidebar_state='auto',
        page_title='Neural Style Transfer',
        page_icon=None,
    )

    available_model_configs = list(SUPPORTED_CONFIGS.keys())
    available_strategies = list(SUPPORTED_FILTERS.keys())

    with st.sidebar:
        st.title('Configuration')
        is_disabled = any(
            [
                not st.session_state.get('selected_content_layer', []),
                not st.session_state.get('selected_style_layers', []),
                not st.session_state.get('content_image'),
                not st.session_state.get('style_image'),
            ],
        )
        st.button('Run NST', on_click=run, disabled=is_disabled)
        model_selection = st.selectbox('Select model', available_model_configs)
        content_table, style_table, optimization_table = st.sidebar.tabs(
            ['Content params', 'Style params', 'Optimization params'],
        )

        with content_table:
            content_file = st.file_uploader(
                'Upload content image', type=['jpg', 'png'],
                key='content_image',
            )
            content_image_path = get_path_from_uploaded(content_file)

            selected_content_layer = st.selectbox(
                'Select layer', get_strategy_params(
                    model_selection,
                ),
                key='selected_content_layer',
            )

        with style_table:
            style_file = st.file_uploader(
                'Upload style image', type=['jpg', 'png'],
                key='style_image',
            )
            style_image_path = get_path_from_uploaded(style_file)
            selected_style_layers = st.multiselect(
                'Select layers', get_strategy_params(
                    model_selection,
                ),
                key='selected_style_layers',
            )

        with optimization_table:
            input_init = st.selectbox(
                'Initialize with', [
                    'random image',
                    'content image', 'style image',
                ],
            )
            content_weight = st.number_input('Content weight', 0., 2., 1., 0.1)
            style_weight = st.number_input('Style weight', 0., 2., 1., 0.1)
            n_iterations = st.number_input('Iterations', 1, 1000, 200, 1)
            regularization_coeff = st.number_input(
                'Regularization coeff', 0., 10., 0.1, 0.05,
            )
            lr = st.number_input('Learning rate', 0.001, 1., 0.3, 0.01)

    images = st.session_state.get('images')
    last_run_params = st.session_state.get('last_run_params')

    if images:
        l_margin, image_col, r_margin = st.columns([1, 3, 1])
        n_images = len(images)
        with l_margin:
            st.write('')
        with image_col:
            img_slider = st.slider('Image slider', 1, n_images, n_images)
            st.image(
                images[img_slider-1], 'Processed Image',
                width=600, use_column_width=True,
            )
        with r_margin:
            st.write('')
        with st.expander('Parameters'):
            params_str = {
                param: str(value)
                for param, value in last_run_params.items()
            }
            st.table(params_str)
