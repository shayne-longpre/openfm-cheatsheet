#!/usr/bin/env python3

"""
To run:

streamlit run ./run_streamlit.py
"""

from datetime import datetime
import json
import numpy as np
import pandas as pd
import math

# from src import util
# from src import filter_util
# from src.helpers import io
import constants
# from src import html_util

import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import streamlit.components.v1 as components
import requests
import webbrowser

from PIL import Image


INFO = {}

@st.cache_data
def load_data():
    return pd.read_csv("resources.csv").fillna("")

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
        filtered_df['Text_Modality'] if text_modality else pd.Series([True] * len(filtered_df)),
        filtered_df['Vision_Modality'] if vision_modality else pd.Series([True] * len(filtered_df)),
        filtered_df['Speech_Modality'] if speech_modality else pd.Series([True] * len(filtered_df))
    ]
    if any([text_modality, vision_modality, speech_modality]):
        filtered_df = filtered_df[pd.concat(modality_conditions, axis=1).any(axis=1)]

    # Apply date filter
    filtered_df = apply_date_filter(filtered_df, time_range)

    return filtered_df


# Load csv --> dataframe --> list of dicts
# filter by section + modality
# present a few rows
# show section introductions
# iterate on showing rows


def add_instructions():
    st.title("Data Provenance Explorer")


def streamlit_app():
    st.set_page_config(page_title="Open Foundation Model Cheatsheet") #, layout="wide")#, initial_sidebar_state='collapsed')

    RESOURCES = load_data()

    st.markdown("""Select the preferred criteria for your datasets.""")


    ### SIDEBAR STARTS HERE

    with st.sidebar:
        
        st.markdown("""Select the preferred criteria for your datasets.""")

        with st.form("data_selection"):

            section_multiselect = st.multiselect(
                'Select the type of resources you are interested in',
                ["All"] + list(set(RESOURCES["Type"])),
                ["All"])

            checkbox_text = st.checkbox("Text", value=True)
            checkbox_vision = st.checkbox("Vision")
            checkbox_speech = st.checkbox("Speech")

            # time_selection = st.slider(
            #     "Select resources from this date onwards",
            #     # options=
            #     value=datetime(2000, 1, 1))

            date_format = 'MMM DD, YYYY'  # format output
            start_date = datetime.date(year=2000,month=1,day=1) #-relativedelta(years=2)  #  I need some range in the past
            end_date = datetime.now().date() #-relativedelta(years=2)
            max_days = end_date-start_date
            
            time_selection = st.slider(
                'Select resources from this date onwards', 
                min_value=start_date, 
                value=start_date,
                max_value=end_date)
                # format=date_format)

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit Selection")

    #### SIDEBAR ENDS HERE


    # st.write(len(INFO["data"]))
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
            st.write(row["Name"] + "  |  " + row["Description"])

        sections = [x for x in constants.ORDERED_SECTION_HEADERS if x in set(filtered_resources["Type"])]
        for section in sections:
            st.header(section)
            st.write(constants.ORDERED_SECTION_HEADERS[section])
            section_resources = filtered_resources[filtered_resources["Type"] == section]
            for row in section_resources:
                write_resource(row)

        # Group sections chronologically.
        # For each section print the title, the subheader (best practices) and each row


if __name__ == "__main__":
    streamlit_app()






    # with st.form("data_selection"):

    #     col1, col2, col3 = st.columns([1,1,1], gap="medium")

    #     with col1:
    #         # st.write("Select the acceptable license values for constituent datasets")
    #         license_multiselect = st.select_slider(
    #             'Select the datasets licensed for these use cases',
    #             options=constants.LICENSE_USE_CLASSES,
    #             value="Academic-Only")

    #         license_attribution = st.toggle('Include Datasets w/ Attribution Requirements', value=True)
    #         license_sharealike = st.toggle('Include Datasets w/ Share Alike Requirements', value=True)
    #         openai_license_override = st.toggle('Always include datasets w/ OpenAI-generated data. (I.e. See `instructions` above for details.)', value=False)

    #     with col3:
            
    #         taskcats_multiselect = st.multiselect(
    #             'Select the task categories to cover in your datasets',
    #             ["All"] + list(INFO["constants"]["TASK_GROUPS"].keys()),
    #             ["All"])

    #     # with st.expander("More advanced criteria"):

    #         # format_multiselect = st.multiselect(
    #         #     'Select the format types to cover in your datasets',
    #         #     ["All"] + INFO["constants"]["FORMATS"],
    #         #     ["All"])

    #         domain_multiselect = st.multiselect(
    #             'Select the domain types to cover in your datasets',
    #             ["All"] + list(INFO["constants"]["DOMAIN_GROUPS"].keys()),
    #             # ["All", "Books", "Code", "Wiki", "News", "Biomedical", "Legal", "Web", "Math+Science"],
    #             ["All"])


    #     with col2:
    #         language_multiselect = st.multiselect(
    #             'Select the languages to cover in your datasets',
    #             ["All"] + list(INFO["constants"]["LANGUAGE_GROUPS"].keys()),
    #             ["All"])

    #         time_range_selection = st.slider(
    #             "Select data release time constraints",
    #             value=(datetime(2000, 1, 1), datetime(2023, 12, 1)))

    #         # st.write("")
    #     # st.write("")
    #     st.divider()

    #     # Every form must have a submit button.
    #     submitted = st.form_submit_button("Submit Selection")

    # #### ALTERNATIVE ENDS HERE