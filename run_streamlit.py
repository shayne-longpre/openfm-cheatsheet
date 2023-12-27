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
# from src import constants
# from src import html_util

import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import streamlit.components.v1 as components
import requests
import webbrowser

from PIL import Image


INFO = {}


# @st.cache_data
# def load_constants():
#     return io.read_all_constants()

# @st.cache_data
# def load_data():
#     data_summary = io.read_data_summary_json("data_summaries/")
#     data_summary = filter_util.map_license_criteria(data_summary, INFO["constants"])
#     return pd.DataFrame(data_summary).fillna("")


# def render_tweet(tweet_url):
#     api = "https://publish.twitter.com/oembed?url={}".format(tweet_url)
#     response = requests.get(api)
#     html_result = response.json()["html"] 
#     st.text(html_result)
#     components.html(html_result, height= 360, scrolling=True)

# def insert_main_viz():

#     # p5.js embed
#     sketch = '<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.6.0/p5.js"></script>'
#     sketch += '<script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.6.0/addons/p5.sound.min.js"></script>'
#     sketch += '<script>'
#     sketch += "const JSONDATA = " + open("static/ds_viz1.json", 'r', encoding='utf-8').read() + "\n"
#     sketch += open("static/sketch.js", 'r', encoding='utf-8').read()
#     sketch += '</script>'
#     components.html(sketch, height=800, scrolling=True)

# def custom_metric(caption, score, delta=None):
#     st.markdown("## :green[" + str(score) + "]")
#     # st.subheader("     :green[" + str(score) + "]")
#     if delta:
#         st.markdown("     " + str(delta))
#     st.markdown(f":gray[{caption}]")
#     # st.caption(caption)

#     # :red[**NOT** to be taken as legal advice]


# def display_metrics(metrics, df_metadata):
#     # metric_columns = st.columns(4)
#     # metric_columns[0].metric("Collections", len(metrics["collections"]), delta=f"/ {len(df_metadata['collections'])}")#, delta_color="off")
#     # metric_columns[1].metric("Datasets", len(metrics["datasets"]), delta=f"/ {len(df_metadata['datasets'])}")
#     # metric_columns[2].metric("Languages", len(metrics["languages"]), delta=f"/ {len(df_metadata['languages'])}")
#     # metric_columns[3].metric("Task Categories", len(metrics["task_categories"]), delta=f"/ {len(df_metadata['task_categories'])}")
#     metric_columns = st.columns(3)
#     # with metric_columns[0]:
#     #     st.metric("Collections", len(metrics["collections"]), delta=f"/ {len(df_metadata['collections'])}")#, delta_color="off")
#     #     st.metric("Datasets", len(metrics["datasets"]), delta=f"/ {len(df_metadata['datasets'])}")
#     #     st.metric("Dialogs", metrics["dialogs"], delta=f"/ {df_metadata['dialogs']}")
#     # with metric_columns[1]:
#     #     st.metric("Languages", len(metrics["languages"]), delta=f"/ {len(df_metadata['languages'])}")
#     #     st.metric("Task Categories", len(metrics["task_categories"]), delta=f"/ {len(df_metadata['task_categories'])}")
#     #     st.metric("Topics", len(metrics["topics"]), delta=f"/ {len(df_metadata['topics'])}")
#     # with metric_columns[2]:
#     #     st.metric("Text Domains", len(metrics["domains"]), delta=f"/ {len(df_metadata['domains'])}")
#     #     st.metric("Text Sources", len(metrics["sources"]), delta=f"/ {len(df_metadata['sources'])}")
#     #     st.metric("% Synthetic Text", metrics["synthetic_pct"])
#     with metric_columns[0]:
#         custom_metric("Collections", len(metrics["collections"]), delta=f"/ {len(df_metadata['collections'])}")#, delta_color="off")
#         custom_metric("Datasets", len(metrics["datasets"]), delta=f"/ {len(df_metadata['datasets'])}")
#         custom_metric("Dialogs", metrics["dialogs"], delta=f"/ {df_metadata['dialogs']}")
#     with metric_columns[1]:
#         custom_metric("Languages", len(metrics["languages"]), delta=f"/ {len(df_metadata['languages'])}")
#         custom_metric("Task Categories", len(metrics["task_categories"]), delta=f"/ {len(df_metadata['task_categories'])}")
#         custom_metric("Topics", len(metrics["topics"]), delta=f"/ {len(df_metadata['topics'])}")
#     with metric_columns[2]:
#         custom_metric("Text Domains", len(metrics["domains"]), delta=f"/ {len(df_metadata['domains'])}")
#         custom_metric("Text Sources", len(metrics["sources"]), delta=f"/ {len(df_metadata['sources'])}")
#         custom_metric("% Synthetic Text", metrics["synthetic_pct"])


def add_instructions():
    st.title("Data Provenance Explorer")

    # col1, col2 = st.columns([0.75, 0.25], gap="medium")

    # with col1:
    #     intro_sents = "The Data Provenance Initiative is a large-scale audit of AI datasets used to train large language models. As a first step, we've traced 1800+ popular, text-to-text finetuning datasets from origin to creation, cataloging their data sources, licenses, creators, and other metadata, for researchers to explore using this tool."
    #     follow_sents = "The purpose of this work is to improve transparency, documentation, and informed use of datasets in AI. "
    #     st.write(" ".join([intro_sents, follow_sents]))
    #     st.write("You can download this data (with filters) directly from the [Data Provenance Collection](https://github.com/Data-Provenance-Initiative/Data-Provenance-Collection).")
    #     st.write("If you wish to contribute or discuss, please feel free to contact the organizers at [data.provenance.init@gmail.com](mailto:data.provenance.init@gmail.com).")
    #     # st.write("NB: This data is compiled voluntarily by the best efforts of academic & independent researchers, and is :red[**NOT** to be taken as legal advice].")

    #     st.write("NB: It is important to note we collect *self-reported licenses*, from the papers and repositories that released these datasets, and categorize them according to our best efforts, as a volunteer research and transparency initiative. The information provided by any of our works and any outputs of the Data Provenance Initiative :red[do **NOT**, and are **NOT** intended to, constitute legal advice]; instead, all information, content, and materials are for general informational purposes only.")

    #     col1a, col1b, col1c = st.columns([0.16, 0.16, 0.68], gap="small")
    #     with col1a:
    #         st.link_button("Data Repository", 'https://github.com/Data-Provenance-Initiative/Data-Provenance-Collection', type="primary")
    #     with col1b:
    #         st.link_button("Paper", 'https://www.dataprovenance.org/paper.pdf', type="primary")

    #     # col1a, col1b = st.columns(2, gap="large")
    #     # with col1a:
    #         # st.link_button("Paper", 'https://www.dataprovenance.org/paper.pdf', type="primary")
    #     # with col1b:
    #         # st.link_button("Data Repository", 'https://github.com/Data-Provenance-Initiative/Data-Provenance-Collection', type="primary")
    #     # st.link_button('Paper', 'https://www.dataprovenance.org/paper.pdf', type="primary")
    #     # st.link_button('Download Repository', 'https://github.com/Data-Provenance-Initiative/Data-Provenance-Collection', type="primary")
    #     # if st.button('Paper', type="primary"):
    #     #     webbrowser.open_new_tab('https://www.dataprovenance.org/paper.pdf')
    #     # if st.button('Download Repository', type="primary"):
    #     #     webbrowser.open_new_tab('https://github.com/Data-Provenance-Initiative/Data-Provenance-Collection')

    #     # URL_STRING = "https://streamlit.io/"

    #     # st.markdown(
    #     #     f'<a href="{URL_STRING}" style="display: inline-block; padding: 12px 20px; background-color: #4CAF50; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 4px;">Action Text on Button</a>',
    #     #     unsafe_allow_html=True
    #     # )
    # with col2:
    #     image = Image.open('dpi.png')
    #     st.image(image)#, caption='Sunrise by the mountains')

    # st.subheader("Instructions")
    # form_instructions = """
    # 1. **Select from the licensed data use cases**. The options range from least to most strict:
    # `Commercial`, `Unspecified`, `Non-Commercial`, `Academic-Only`.
    
    # * `Commercial` will select only the data with licenses explicitly permitting commercial use. 
    # * `Unspecified` includes Commercial plus datasets with no license found attached, which may suggest the curator does not prohibit commercial use.
    # * `Non-Commercial` includes Commercial and Unspecified datasets plus those licensed for non-commercial use.
    # * `Academic-Only` will select all available datasets, including those that restrict to only academic uses.

    # Note that these categories reflect the *self-reported* licenses attached to datasets, and assume fair use of any data they are derived from (e.g. scraped from the web).

    # 2. Select whether to include datasets with **Attribution requirements in their licenses**.

    # 3. Select whether to include datasets with **`Share-Alike` requirements in their licenses**. 
    # Share-Alike means a copyright left license, that allows other to re-use, re-mix, and modify works, but requires that derivative work is distributed under the same terms and conditions.

    # 4. Select whether to ignore the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use) as a Non-Commercial restriction, and include datasets that are at least partially **generated by OpenAI** (inputs, outputs, or both).
    # While the OpenAI terms state you cannot ``use output from the Services to develop models that compete with OpenAI'', there is debate as to their enforceability and applicability to third parties who did not generate this data themselves. See our Legal Discussion section in the paper for more discussion on these terms.
    
    # 5. **Select Language Families** to include.

    # 6. **Select Task Categories** to include.

    # 7. **Select Time of Collection**. By default it includes all datasets.

    # 8. **Select the Text Domains** to include.

    # Finally, Submit Selection when ready!
    # """
    # with st.expander("Expand for Instructions!"):
    #     st.write(form_instructions)

def streamlit_app():
    st.set_page_config(page_title="Data Provenance Explorer", layout="wide")#, initial_sidebar_state='collapsed')
    # INFO["constants"] = load_constants()
    # st.write(INFO["constants"].keys())
    # INFO["data"] = load_data()

    # df_metadata = util.compute_metrics(INFO["data"], INFO["constants"])

    # add_instructions()

    #### ALTERNATIVE STARTS HERE
    st.markdown("""Select the preferred criteria for your datasets.""")


if __name__ == "__main__":
    streamlit_app()
