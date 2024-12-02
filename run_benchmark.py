#!/usr/bin/env python3

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Benchmark AI tool against ground truth data.')
    parser.add_argument('--output_dir', required=True, help='Path to the AI output directory.')
    parser.add_argument('--ground_truth_dir', default='data/ground_truth', help='Path to the ground truth directory (default: data/ground_truth).')
    parser.add_argument('--excel_file', default='data/benchmark_output/benchmark_results.xlsx', help='Output Excel file name (default: benchmark_results.xlsx).')

    args = parser.parse_args()
    output_dir = args.output_dir
    ground_truth_dir = args.ground_truth_dir
    excel_file = args.excel_file

    # Initialize lists to store metrics for all manuscripts
    figure_detection_recalls = []
    figure_detection_precisions = []
    figure_caption_rouge_scores = []
    panel_caption_rouge_scores = []
    panel_label_accuracies = []
    duplicate_panel_labels_counts = []

    # Get list of JSON files in the output directory
    output_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]

    # Initialize a list to collect all results
    results_list = []

    # Function to compute ROUGE-L score
    def compute_rouge_l_score(reference, hypothesis):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores['rougeL'].fmeasure

    # Regular expression to parse filenames
    filename_pattern = re.compile(r'^output_(?P<manuscript_id>[^_]+)_(?P<non_standard_text>.*)_(?P<iteration>\d+)\.json$')

    for output_file in output_files:
        # Extract manuscript_id and iteration from filename
        match = filename_pattern.match(output_file)
        if not match:
            print(f"Filename {output_file} does not match the expected pattern. Skipping.")
            continue

        manuscript_id = match.group('manuscript_id')
        iteration = match.group('iteration')

        # Paths to the AI output and ground truth JSON files
        ai_output_path = os.path.join(output_dir, output_file)
        ground_truth_filename = f"{manuscript_id}.json"
        ground_truth_path = os.path.join(ground_truth_dir, ground_truth_filename)

        # Check if ground truth file exists
        if not os.path.exists(ground_truth_path):
            print(f"Ground truth missing for manuscript {manuscript_id}. Skipping.")
            continue

        # Load JSON data
        try:
            with open(ai_output_path, 'r') as f:
                ai_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding AI output JSON for manuscript {manuscript_id}, iteration {iteration}: {e}")
            continue

        try:
            with open(ground_truth_path, 'r') as f:
                gt_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding ground truth JSON for manuscript {manuscript_id}: {e}")
            continue

        # Extract figures
        ai_figures = ai_data.get('figures', [])
        gt_figures = gt_data.get('figures', [])

        # Compute figure detection metrics
        total_gt_figures = len(gt_figures)
        total_ai_figures = len(ai_figures)
        correctly_detected_figures = 0

        ai_figure_labels = set(fig.get('figure_label', '') for fig in ai_figures)
        gt_figure_labels = set(fig.get('figure_label', '') for fig in gt_figures)
        correctly_detected_figures = len(ai_figure_labels & gt_figure_labels)

        recall = correctly_detected_figures / total_gt_figures if total_gt_figures else 0
        precision = correctly_detected_figures / total_ai_figures if total_ai_figures else 0

        figure_detection_recalls.append(recall)
        figure_detection_precisions.append(precision)

        # Compute ROUGE-L scores for figure captions
        figure_caption_scores = []
        for gt_fig in gt_figures:
            gt_label = gt_fig.get('figure_label', '')
            gt_caption = gt_fig.get('figure_caption', '')

            # Find the matching AI figure
            ai_fig = next((fig for fig in ai_figures if fig.get('figure_label', '') == gt_label), None)
            if ai_fig:
                ai_caption = ai_fig.get('figure_caption', '')
                score = compute_rouge_l_score(gt_caption, ai_caption)
                figure_caption_scores.append(score)

        avg_figure_caption_rouge = np.mean(figure_caption_scores) if figure_caption_scores else 0
        figure_caption_rouge_scores.append(avg_figure_caption_rouge)

        # Initialize per-manuscript variables
        panel_caption_scores = []
        correct_panel_labels = 0
        total_gt_panels = 0
        duplicate_panel_labels = 0

        for gt_fig in gt_figures:
            gt_label = gt_fig.get('figure_label', '')
            gt_panels = gt_fig.get('panels', [])
            total_gt_panels += len(gt_panels)

            # Find the matching AI figure
            ai_fig = next((fig for fig in ai_figures if fig.get('figure_label', '') == gt_label), None)
            if ai_fig:
                ai_panels = ai_fig.get('panels', [])
                ai_panels_dict = defaultdict(list)
                ai_panel_labels = []  # Collect panel labels for this figure

                for panel in ai_panels:
                    ai_panel_label = panel.get('panel_label', '')
                    ai_panels_dict[ai_panel_label].append(panel)
                    ai_panel_labels.append(ai_panel_label)

                # Check for duplicates within this figure
                label_counts = pd.Series(ai_panel_labels).value_counts()
                duplicates_in_figure = (label_counts[label_counts > 1] - 1).sum()
                duplicate_panel_labels += duplicates_in_figure

                for gt_panel in gt_panels:
                    gt_panel_label = gt_panel.get('panel_label', '')
                    gt_panel_caption = gt_panel.get('panel_caption', '')

                    # Find matching AI panel(s)
                    ai_panels_list = ai_panels_dict.get(gt_panel_label, [])
                    if ai_panels_list:
                        correct_panel_labels += 1
                        # Take the first matching panel (could be extended to average over all)
                        ai_panel = ai_panels_list[0]
                        ai_panel_caption = ai_panel.get('panel_caption', '')
                        score = compute_rouge_l_score(gt_panel_caption, ai_panel_caption)
                        panel_caption_scores.append(score)

        duplicate_panel_labels_counts.append(duplicate_panel_labels)

        panel_label_accuracy = correct_panel_labels / total_gt_panels if total_gt_panels else 0
        panel_label_accuracies.append(panel_label_accuracy)

        avg_panel_caption_rouge = np.mean(panel_caption_scores) if panel_caption_scores else 0
        panel_caption_rouge_scores.append(avg_panel_caption_rouge)

        # Collect results for this manuscript and iteration
        manuscript_results = {
            'manuscript_id': manuscript_id,
            'iteration': iteration,
            'figure_recall': recall,
            'figure_precision': precision,
            'avg_figure_caption_rouge': avg_figure_caption_rouge,
            'panel_label_accuracy': panel_label_accuracy,
            'avg_panel_caption_rouge': avg_panel_caption_rouge,
            'duplicate_panel_labels': duplicate_panel_labels,
        }
        # Append the manuscript results to the list
        results_list.append(manuscript_results)

    if not results_list:
        print("No valid data to process. Exiting.")
        sys.exit(1)

    # Create the DataFrame from the list of results
    results_df = pd.DataFrame(results_list)

    # Compute overall statistics
    overall_metrics = {
        'figure_recall_mean': np.mean(figure_detection_recalls),
        'figure_recall_std': np.std(figure_detection_recalls),
        'figure_precision_mean': np.mean(figure_detection_precisions),
        'figure_precision_std': np.std(figure_detection_precisions),
        'avg_figure_caption_rouge_mean': np.mean(figure_caption_rouge_scores),
        'avg_figure_caption_rouge_std': np.std(figure_caption_rouge_scores),
        'panel_label_accuracy_mean': np.mean(panel_label_accuracies),
        'panel_label_accuracy_std': np.std(panel_label_accuracies),
        'avg_panel_caption_rouge_mean': np.mean(panel_caption_rouge_scores),
        'avg_panel_caption_rouge_std': np.std(panel_caption_rouge_scores),
        'duplicate_panel_labels_total': np.sum(duplicate_panel_labels_counts),
    }

    # Output the results
    print("Per Manuscript Results:")
    print(results_df)

    print("\nOverall Metrics:")
    for metric, value in overall_metrics.items():
        print(f"{metric}: {value}")

    # Save results to Excel, adding a new sheet named as the output directory
    sheet_name = os.path.basename(os.path.normpath(output_dir))
    # Replace invalid characters in sheet name
    sheet_name = re.sub(r'[\\/*?:\[\]]', '_', sheet_name)
    # Check if Excel file exists
    if os.path.exists(excel_file):
        # Open the existing Excel file
        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
            # Remove sheet if it already exists
            if sheet_name in writer.book.sheetnames:
                del writer.book[sheet_name]
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # Create a new Excel file
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    # Save results to Excel, adding a new sheet named as the output directory
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
        # Remove sheet if it already exists
        if f"{sheet_name}_per_manuscript" in writer.book.sheetnames:
            del writer.book[f"{sheet_name}_per_manuscript"]
        results_df.to_excel(writer, sheet_name=f"{sheet_name}_per_manuscript", index=False)


    print(f"\nResults have been saved to {excel_file} in sheet '{sheet_name}'.")

if __name__ == '__main__':
    main()
