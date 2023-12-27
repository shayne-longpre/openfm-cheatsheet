#!/usr/bin/env python3

"""
To run:

streamlit run ./run_streamlit.py
"""

import datetime as dt
# from datetime import datetime
import json
import numpy as np
import pandas as pd
import math

import constants

import streamlit as st
# from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
# import streamlit.components.v1 as components
import requests
import webbrowser

from PIL import Image
import base64


INFO = {}

@st.cache_data
def load_data():
    return pd.read_csv("resources.csv").fillna("")

def load_logos():
    def get_image_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return {
        "hf": get_image_base64("logos/web.png"),
        "web": get_image_base64("logos/hf.png"),
        "arxiv": get_image_base64("logos/arxiv.png"),
        "github": get_image_base64("logos/github.png"),
    }

def is_date_match(release_date, filter_start, filter_end="2030"):
    def convert_to_dt(x):
        if isinstance(x, str):
            return pd.to_datetime(x, format='%m-%Y') if "-" in x else pd.to_datetime(x, format='%Y')
        return x

    if not release_date or release_date.lower() == "frequently updated":
        return True

    try:
        release_datetime = convert_to_dt(release_date)
        filter_start_dt = convert_to_dt(filter_start)
        filter_end_dt = convert_to_dt(filter_end)

        return filter_start_dt <= release_datetime <= filter_end_dt
    except Exception as e:
        raise ValueError(f"Incorrect date format: {e}")

# Function to apply the date filter
def apply_date_filter(df, release_date):
    return df[df['Release MM-YY'].apply(lambda x: is_date_match(x, release_date))]


# Function to search for resources
def search_resources(df, search_string):
    return df[df.apply(lambda row: search_string.lower() in row['Name'].lower() or search_string.lower() in row['Description'].lower(), axis=1)]


def preprocess_modalities(df):
    df['Text_Modality'] = df['Modalities'].str.contains('Text') | (df['Modalities'] == 'All')
    df['Vision_Modality'] = df['Modalities'].str.contains('Vision') | (df['Modalities'] == 'All')
    df['Speech_Modality'] = df['Modalities'].str.contains('Speech') | (df['Modalities'] == 'All')
    return df

def filter_resources(
    resources_df, 
    sections, 
    text_modality,
    vision_modality,
    speech_modality,
    time_range
):
    
    # Preprocess the DataFrame to add modality columns
    filtered_df = preprocess_modalities(resources_df)

    # Apply sections filter
    if "All" not in sections:
        filtered_df = filtered_df[filtered_df['Type'].isin(sections)]

    # Apply combined modality filter using any
    modality_conditions = [
        filtered_df['Text_Modality'] if text_modality else pd.Series([False] * len(filtered_df)),
        filtered_df['Vision_Modality'] if vision_modality else pd.Series([False] * len(filtered_df)),
        filtered_df['Speech_Modality'] if speech_modality else pd.Series([False] * len(filtered_df))
    ]
    if any([text_modality, vision_modality, speech_modality]):
        filtered_df = filtered_df[pd.concat(modality_conditions, axis=1).any(axis=1)]

    # Apply date filter
    filtered_df = apply_date_filter(filtered_df, time_range)

    return filtered_df


def streamlit_app():
    st.set_page_config(page_title="Open Foundation Model Cheatsheet", layout="wide") #, initial_sidebar_state='collapsed')

    RESOURCES = load_data()
    LOGOS = load_logos()

    st.title("Open Foundation Model Cheatsheet")
    st.caption("Resources and recommendations for best practices in developing and releasing open models.")
    st.markdown("""This cheatsheet serves as a succinct guide, prepared *by* open foundation model developers *for* developers.
        As AI foundation model development rapidly expands, welcoming new contributors, scientists, and
        applications, we hope to help new community members become familiar with the latest resources, tools, and growing
        body of research findings. The focus of this cheatsheet is not only to support building, but also to inculcate good
        practices, awareness of limitations, and general responsible habits as community norms.""")
    scope_limitations_text = """
        There are many exceedingly popular tools to build, distribute and deploy foundation models. But there
        are also many incredible resources that have gone less noticed, in the community's efforts to accelerate, deploy, and
        monetize. We hope to bring wider attention to these core resources that support informed data selection, processing,
        and understanding, precise and limitation-aware artifact documentation, resource-frugal model
        training, advance awareness of the environmental impact from training, careful model evaluation
        and claims, and lastly, responsible model release and deployment practices.

        We've compiled strong resources, tools, and papers that have helped
        guide our own intuitions around model development, and which we believe will be especially helpful to nascent
        (and sometimes even experienced) developers in the field. However, this guide is certainly not comprehensive or
        perfect—and here's what to consider when using it:
        * They are scoped to 'open' model development, by which we mean the model weights will be publicly downloadable.
        * Foundation model development is a rapidly evolving science. This cheatsheet is dated to December 2023, but
        our repository is open for on-going public contributions: www.comingsoon.com.
        * We've scoped our content modalities only to text, vision, and speech. We attempt to support multilingual
        resources, but acknowledge these are only a starting point.
        * A cheatsheet cannot be comprehensive. We rely heavily on survey papers and repositories to point out the
        many other awesome works which deserve consideration, especially for developers who plan to dive deeper
        into a topic.
        * Lastly, we do not recommend all these resources for all circumstances, and have provided notes throughout
        to guide this judgement. Instead we hope to bring awareness to good practices that many developers neglect
        in the haste of development (eg careful data decontamination, documentation, and carefully specifying the
        intended downstream uses).
    """
    with st.expander("Scope & Limitations"):
        st.markdown(scope_limitations_text)
    col1a, col1b, col1c = st.columns([0.3, 0.3, 0.4], gap="small")
    # TODO: Replace button links.
    with col1a:
        st.link_button("\"2023 Wrapped\" Cheatsheet Paper", 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', type="primary")
    with col1b:
        st.link_button("Contribute Resources!", 'https://www.youtube.com/watch?v=dQw4w9WgXcQ', type="primary")
    st.markdown("""Assembled by open model developers from AI2, EleutherAI, Google, Hugging Face, Masakhane, McGill, MIT, Princeton, Stanford CRFM, and UCSB.""")

    ### SIDEBAR STARTS HERE

    with st.sidebar:
        
        st.markdown("""Select the resource criteria.""")

        with st.form("data_selection"):

            section_multiselect = st.multiselect(
                label='Resource Types:',
                options=["All"] + list(set(RESOURCES["Type"])),
                default=["All"])

            st.markdown("####")
            # st.divider()

            st.markdown('<p style="font-size: 14px;">Modality Types:</p>', unsafe_allow_html=True)

            # st.markdown("Modality Types:")
            checkbox_text = st.checkbox("Text", value=True)
            checkbox_vision = st.checkbox("Vision")
            checkbox_speech = st.checkbox("Speech")

            st.markdown("####")
            # st.divider()

            date_format = 'MMM, YYYY'  # format output
            start_date = dt.date(year=2000,month=1,day=1)
            end_date = dt.datetime.now().date()
            max_days = end_date-start_date
            
            time_selection = st.slider(
                label="Start Date:", 
                min_value=start_date, 
                value=start_date,
                max_value=end_date,
                format=date_format)

            st.markdown("####")

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit Selection")


    if submitted:
        filtered_resources = filter_resources(
            RESOURCES, 
            sections=section_multiselect, 
            text_modality=checkbox_text,
            vision_modality=checkbox_vision,
            speech_modality=checkbox_speech,
            time_range=time_selection
        )

        def write_resource(row):
            # TODO: Hyperlinks and modalities on each row.
            col1, col2, col3 = st.columns([1,4,1], gap="small")
            col1.write(row["Name"])
            col2.write(row["Description"])

            def create_markdown_img(base64_string, link_url, dim=30):
                img_tag = f'<img src="data:image/png;base64,{base64_string}" width="{dim}" height="{dim}" alt="Image">'
                return f'<a href="{link_url}" target="_blank">{img_tag}</a>'

            # # URL of the hyperlink
            # link_url = "https://dataprovenance.org"

            for logo_img, col in [
                (LOGOS['arxiv'], 'Paper Link'), 
                (LOGOS['hf'], 'HF Link'),
                (LOGOS['github'], 'GitHub Link'), 
                (LOGOS['web'], 'Website Link'), 
            ]:
                if row[col]:
                    col3.markdown(create_markdown_img(logo_img, row[col]), unsafe_allow_html=True)
                else:
                    continue
                    # col3.markdown(f"<div style='width: 1px;'></div>", unsafe_allow_html=True)

                
                

            #     # Create the markdown string with the base64 image and link
            #     markdown_string = create_markdown_img(base64_string, link_url)
            #     


        sections = [x for x in constants.ORDERED_SECTION_HEADERS if x in set(filtered_resources["Type"])]
        for section in sections:
            st.header(section)
            # TODO: show section introductions
            # st.write(constants.ORDERED_SECTION_HEADERS[section])
            st.divider()
            section_resources = filtered_resources[filtered_resources["Type"] == section]
            for i, row in section_resources.iterrows():
                write_resource(row)
                st.divider()

# TODO: Links to paper and submit form (make).
# TODO: Update section names
# TODO: Cheatsheet Logo





if __name__ == "__main__":
    streamlit_app()



