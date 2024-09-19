import streamlit as st
import os
from abc import ABC, abstractmethod

from src.views import View
from src.validation_view import ValidationView

# Custom CSS
custom_css = """
<style>
    .folder-name { font-weight: bold; }
    .file-name { font-size: 0.9em; margin-left: 10px; }
    .icon { font-size: 1.2em; margin-right: 5px; }
    .figure-title { font-size: 1.1em; font-style: italic; color: #555; margin-top: 5px; margin-bottom: 15px; }
</style>
"""


class Router:
    def __init__(self):
        self.routes = {}

    def add_route(self, path, view):
        self.routes[path] = view

    def render(self):
        path = st.session_state.get('path', '/')
        view = self.routes.get(path)
        if view:
            view.render()
        else:
            st.error(f"No view found for path: {path}")


class MainView(View):
    def render(self):
        st.title("Scientific Figure Viewer")

        # Metadata
        st.header("Manuscript: MSB-22-1234")
        st.subheader("Spatially distinct epithelial and mesenchymal cell subsets along progressive lineage restriction in the branching embryonic mammary gland")
        st.text("Authors: Jim Smith, Hannah Sonntag, Korg Hag")

        # Figures as tabs
        figures = [
            ("Figure 1", "Differential gene expression analysis of cells treated with NSF for 4 or 6 h."),
            ("Figure 2", "Protein interaction network analysis."),
            ("Figure 3", "Cellular localization of key proteins."),
            ("Figure 4", "Time-course analysis of gene expression changes."),
            ("Figure 5", "Pathway enrichment analysis."),
        ]
        tabs = st.tabs([fig[0] for fig in figures])

        for i, tab in enumerate(tabs):
            with tab:
                self.display_figure_content(figures[i][0], figures[i][1])

        # Source data associated with the entire paper
        st.header("Source data associated with the entire paper")
        whole_paper_data = [
            {"Whole Paper": [
                "sourcedata_full_paper.zip",
                {"Supplementary Data": [
                    "supplement1.pdf",
                    "supplement2.xlsx"
                ]}
            ]}
        ]
        self.display_hierarchical_data(whole_paper_data)

    def display_hierarchical_data(self, data, indent=0):
        for item in data:
            if isinstance(item, dict):
                for folder_name, contents in item.items():
                    st.markdown(f'<div style="margin-left: {indent * 20}px;"><span class="icon">📁</span><span class="folder-name">{folder_name}</span></div>', unsafe_allow_html=True)
                    self.display_hierarchical_data(contents, indent + 1)
            else:
                st.markdown(f'<div style="margin-left: {indent * 20}px;"><span class="icon">📄</span><span class="file-name">{item}</span></div>', unsafe_allow_html=True)

    def display_figure_content(self, figure_name, figure_title):
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Validate", key=f"validate_{figure_name}", type="primary"):
                st.session_state.path = f'/validate'
                st.rerun()
        with col2:
            st.header(figure_name, divider="red")

        st.markdown(f'<p class="figure-title">{figure_title}</p>', unsafe_allow_html=True)

        col_image, col_data = st.columns([4, 6])

        with col_image:
            image_path = os.path.join(os.path.dirname(__file__), "static", "example.jpg")
            st.image(image_path, caption=f"Image for {figure_name}", use_column_width=True)

        with col_data:
            st.subheader("Associated Data Files")
            figure_data = [
                {"Whole Figure": ["sourcedata_fig1.zip"]},
                {"Panel A": ["filename_1.jpg", "filename_2.jpg", "filename_3.xls", "pride.project:PXD000440"]},
                {"Panel B": ["file1.txt", "file2.csv", {"Subfolder": ["subfile1.png", "subfile2.dat"]}]},
                {"Panel C": ["data.xlsx"]}
            ]
            self.display_hierarchical_data(figure_data)


def main():
    st.markdown(custom_css, unsafe_allow_html=True)

    if 'path' not in st.session_state:
        st.session_state.path = '/'

    router = Router()
    router.add_route('/', MainView())
    router.add_route('/validate', ValidationView())
    router.render()

if __name__ == "__main__":
    main()