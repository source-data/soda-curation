import pandas as pd
import streamlit as st
from src.views import View

# Custom CSS
custom_css = """
<style>
    span.caption {
        font-size: 9px;
    }
    .icon {
        font-size: 1.2em;
        margin-right: 5px;
    }
    .panel-container {
        display: flex;
        align-items: start;
        margin-bottom: 20px;
    }
    .thumbnail {
        width: 150px;
        margin-right: 20px;
    }
    .panel-content {
        flex-grow: 1;
    }
</style>
"""


class ValidationView(View):
    def render(self):
        # st.set_page_config(layout="wide")
        st.markdown(custom_css, unsafe_allow_html=True)

        # figure_name = st.session_state.path.split('/')[-1]
        # st.title(f"Validation View: {figure_name}")
        # st.write("This is the single figure validation view.")
        # st.write("Here you can add validation-specific content for the figure.")

        # if st.button("Back to Main View"):
        #     st.session_state.path = '/'
        #     st.rerun()

        fig_num = "1"
        st.title("Scientific Figure Viewer")
        # Main content
        st.subheader(
            f"Figure {fig_num}. Spatially distinct epithelial and mesenchymal cell subsets along progressive lineage restriction in the branching embryonic mammary gland"
        )

        col1, col2 = st.columns(2)
        with col1:
            st.caption(
                """<span class="caption">(A) Inducible expression of human kinases from a genomic landing pad in S. cerevisiae, 
    followed by data-independent acquisition (DIA) mass spectrometry. WT and kinase-dead mutants for 
    31 kinases, as well as 13 v-SRC variant mutants and controls, were grown in five biological 
    replicates each (n = 390, one failed, three excluded, see Methods). After phosphoproteomics, 
    the impact of phosphorylation on protein structure, fitness, and evolution is analysed. pY: 
    full-length tyrosine kinase, pYd: tyrosine kinase domain, v-SRC: WT v-SRC and its mutants, 
    pS/pT: full-length serine and threonine kinases. (B) Correlated phosphorylation profiles 
    (Pearson's correlation coefficient) between all kinases tested (WT and v-SRC variants), 
    based upon the median phosphosite intensity (pS/pT/pY) across replicates. (C) Number of up- 
    and downregulated pY (dark grey) and pS/pT (light grey) sites per kinase. Up- and 
    downregulation for each WT kinase is with respect to the kinase-dead mutant. (D) Separation 
    of phosphorylation profile ratios (WT/variant vs kinase-dead, n = 226) in two dimensions 
    using the tSNE dimensionality-reduction method. (E) Relative phosphosite abundance log2 
    (WT/dead) for each kinase, with respect to the phosphoacceptor identity (S, T, Y) and whether 
    or not the phosphosite is a member of the core phosphoproteome in S. cerevisiae that is found 
    to be phosphorylated in many conditions (Leutert et al, 2023). Source data are available online 
    for this figure.</span>""",
                unsafe_allow_html=True,
            )

        with col2:
            # Placeholder image
            st.image("https://via.placeholder.com/400x300.png?text=Figure+Placeholder")

        # Data files section
        st.header("Associated Data Files")

        # Whole figure data files
        st.subheader("Whole Figure Data")
        whole_figure_data = [
            {
                "file": "sourcedata_fig1A.zip",
                "description": "Gastrocnemius sections in wild type and mutant ihpr-3",
                "type": "Raw Data",
            },
            {
                "file": "pride.project:PXD000440",
                "description": "Peptide hormone profiling",
                "type": "Mass Spectrometry",
            },
            {
                "file": "metabolights:MTBLS1",
                "description": "Profiling of circulating metabolites",
                "type": "Metabolomics",
            },
        ]
        display_data_files(whole_figure_data, "whole_figure")

        # Panel-specific data files
        panels = [
            {
                "label": "A",
                "thumbnail": "https://via.placeholder.com/150x150.png?text=Panel+A",
                "caption": "Inducible expression of human kinases from a genomic landing pad in S. cerevisiae, followed by data-independent acquisition (DIA) mass spectrometry.",
                "data": [
                    {
                        "file": "sourcedata_fig1A.xlsx",
                        "description": "Western blot",
                        "type": "Quantitiative Data",
                    },
                    {
                        "file": "sourcedata_fig1A.tif",
                        "description": "Microscopy image",
                        "type": "Ligth microscopy",
                    },
                    {
                        "file": "pride.project:PXD00044A",
                        "description": "Mass spectrometry data",
                        "type": "Mass Spectrometry",
                    },
                ],
            },
            {
                "label": "B",
                "thumbnail": "https://via.placeholder.com/150x150.png?text=Panel+B",
                "caption": "Correlated phosphorylation profiles (Pearson's correlation coefficient) between all kinases tested (WT and v-SRC variants), based upon the median phosphosite intensity (pS/pT/pY) across replicates.",
                "data": [
                    {
                        "file": "sourcedata_fig1B.xlsx",
                        "description": "Correlation data",
                        "type": "Quantitiative Data",
                    },
                ],
            },
            # Add more panels here...
        ]

        for panel in panels:
            st.markdown(
                f"""
            <div class="panel-container">
                <img src="{panel['thumbnail']}" class="thumbnail">
                <div class="panel-content">
                    <h3>Panel {panel['label']}</h3>
                    <p>{panel['caption']}</p>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            display_data_files(panel["data"], f"panel_{panel['label']}")


def display_data_files(data_files, key):
    if key not in st.session_state:
        df = pd.DataFrame(data_files)
        st.session_state[key] = df
    else:
        df = st.session_state[key]

    # Define the options for the data type dropdown
    data_type_options = [
        "Quantitiative Data",
        "Blots and gels",
        "Ligth microscopy",
        "EM",
        "CryoEM",
        "Sequencing ",
        "Mass Spectrometry",
        "Metabolomics",
        "FACS",
        "Other",
    ]

    # Create a new DataFrame with the dropdown for data type and checkbox for removal
    edited_df = st.data_editor(
        df,
        column_config={
            "file": "File",
            "description": "Description",
            "type": st.column_config.SelectboxColumn(
                "Data Type", options=data_type_options, required=True
            ),
        },
        num_rows="dynamic",
        key=f"data_editor_{key}",
    )

    # Add file button
    if st.button("Add file or link", key=f"add_file_{key}"):
        new_row = pd.DataFrame(
            [{"file": "", "description": "", "type": "Quantitiative Data"}]
        )
        edited_df = pd.concat([edited_df, new_row], ignore_index=True)

    # Update the session state
    st.session_state[key] = edited_df

    return edited_df
