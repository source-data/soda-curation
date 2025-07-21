#!/usr/bin/env python3
"""
QC Analysis tool for comparing QC results with actual figure content.

This script analyzes QC results against the extracted figure images and captions
to identify potential issues with the AI's analysis.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .data_storage import load_figure_data

logger = logging.getLogger(__name__)


class QCAnalysis:
    """Analyze QC results against actual figure content."""

    def __init__(self):
        """Initialize the QC analyzer."""
        pass

    def analyze_qc_results(
        self, qc_results_file: str, figure_data_file: str, output_report: str = None
    ) -> Dict[str, Any]:
        """
        Compare QC results with actual figure data and identify issues.

        Args:
            qc_results_file: Path to QC results JSON
            figure_data_file: Path to figure data JSON
            output_report: Optional path to save detailed report

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Loading QC results from: {qc_results_file}")
        with open(qc_results_file, "r") as f:
            qc_results = json.load(f)

        logger.info(f"Loading figure data from: {figure_data_file}")
        figure_data = load_figure_data(figure_data_file)

        # Convert figure data to dictionary for easier lookup
        figure_dict = {
            label: (image_data, caption) for label, image_data, caption in figure_data
        }

        analysis_results = {
            "summary": {
                "total_figures": len(figure_data),
                "figures_analyzed": len(qc_results.get("figures", {})),
                "total_panels_detected": 0,
                "issues_found": 0,
                "critical_issues": 0,
            },
            "figure_analysis": {},
            "issues": [],
            "recommendations": [],
        }

        logger.info("Starting detailed QC analysis...")

        # Analyze each figure
        for figure_id, qc_figure_data in qc_results.get("figures", {}).items():
            figure_label = figure_id.replace(
                "_", " "
            ).title()  # Convert figure_1 -> Figure 1

            if figure_label not in figure_dict:
                logger.warning(f"QC result {figure_id} not found in figure data")
                continue

            image_data, caption = figure_dict[figure_label]

            # Analyze this figure
            figure_analysis = self._analyze_figure(
                figure_id, figure_label, qc_figure_data, caption
            )

            analysis_results["figure_analysis"][figure_id] = figure_analysis
            analysis_results["summary"]["total_panels_detected"] += len(
                qc_figure_data.get("panels", [])
            )
            analysis_results["summary"]["issues_found"] += len(
                figure_analysis["issues"]
            )
            analysis_results["summary"]["critical_issues"] += len(
                figure_analysis["critical_issues"]
            )

            # Add issues to global list
            for issue in figure_analysis["issues"]:
                analysis_results["issues"].append(
                    {"figure": figure_id, "severity": "warning", "issue": issue}
                )

            for issue in figure_analysis["critical_issues"]:
                analysis_results["issues"].append(
                    {"figure": figure_id, "severity": "critical", "issue": issue}
                )

        # Generate recommendations
        analysis_results["recommendations"] = self._generate_recommendations(
            analysis_results
        )

        # Save detailed report if requested
        if output_report:
            self._save_detailed_report(analysis_results, output_report)

        logger.info(
            f"QC analysis complete. Found {analysis_results['summary']['issues_found']} issues."
        )

        return analysis_results

    def _analyze_figure(
        self, figure_id: str, figure_label: str, qc_data: Dict[str, Any], caption: str
    ) -> Dict[str, Any]:
        """Analyze a single figure's QC results."""
        analysis = {
            "figure_id": figure_id,
            "figure_label": figure_label,
            "panels_detected": len(qc_data.get("panels", [])),
            "issues": [],
            "critical_issues": [],
            "panel_analysis": {},
            "test_results": {},
        }

        # Analyze panels
        panels = qc_data.get("panels", [])
        for panel in panels:
            panel_label = panel.get("panel_label", "unknown")
            panel_analysis = self._analyze_panel(panel, caption, figure_id)
            analysis["panel_analysis"][panel_label] = panel_analysis

            # Collect issues from panel analysis
            analysis["issues"].extend(panel_analysis["issues"])
            analysis["critical_issues"].extend(panel_analysis["critical_issues"])

        # Check for common figure-level issues
        self._check_figure_level_issues(analysis, caption)

        return analysis

    def _analyze_panel(
        self, panel_data: Dict[str, Any], caption: str, figure_id: str
    ) -> Dict[str, Any]:
        """Analyze a single panel's QC results."""
        panel_label = panel_data.get("panel_label", "unknown")
        qc_tests = panel_data.get("qc_tests", [])

        analysis = {
            "panel_label": panel_label,
            "total_tests": len(qc_tests),
            "passed_tests": 0,
            "failed_tests": 0,
            "issues": [],
            "critical_issues": [],
            "test_details": {},
        }

        # Analyze each test
        for test in qc_tests:
            test_name = test.get("test_name", "unknown")
            passed = test.get("passed", False)
            model_output = test.get("model_output", {})

            if passed:
                analysis["passed_tests"] += 1
            else:
                analysis["failed_tests"] += 1

            # Analyze specific test results
            test_analysis = self._analyze_test_result(test_name, model_output, caption)
            analysis["test_details"][test_name] = test_analysis

            # Check for potential issues in the test results
            issues = self._check_test_issues(
                test_name, model_output, caption, panel_label
            )
            analysis["issues"].extend(issues["warnings"])
            analysis["critical_issues"].extend(issues["critical"])

        return analysis

    def _analyze_test_result(
        self, test_name: str, model_output: Dict[str, Any], caption: str
    ) -> Dict[str, Any]:
        """Analyze a specific test result."""
        return {
            "test_name": test_name,
            "model_interpretation": model_output,
            "caption_relevance": self._check_caption_relevance(
                test_name, model_output, caption
            ),
        }

    def _check_caption_relevance(
        self, test_name: str, model_output: Dict[str, Any], caption: str
    ) -> Dict[str, Any]:
        """Check if the model's interpretation aligns with the caption content."""
        relevance = {"alignment": "unknown", "confidence": 0.0, "notes": []}

        # Check specific test types
        if test_name == "stat_test":
            statistical_test_needed = (
                model_output.get("statistical_test_needed", "").lower() == "yes"
            )

            # Look for statistical indicators in caption
            stat_indicators = [
                "p <",
                "p=",
                "p-value",
                "statistical",
                "significant",
                "ANOVA",
                "t-test",
                "chi-square",
                "mann-whitney",
                "wilcoxon",
                "±",
                "mean ±",
            ]
            has_stats_in_caption = any(
                indicator.lower() in caption.lower() for indicator in stat_indicators
            )

            if has_stats_in_caption and not statistical_test_needed:
                relevance["alignment"] = "misaligned"
                relevance["notes"].append(
                    "Caption mentions statistical analysis but model says no test needed"
                )
            elif not has_stats_in_caption and statistical_test_needed:
                relevance["alignment"] = "questionable"
                relevance["notes"].append(
                    "Model says statistical test needed but none mentioned in caption"
                )
            else:
                relevance["alignment"] = "aligned"

        elif test_name == "error_bars_defined":
            error_bar_on_figure = (
                model_output.get("error_bar_on_figure", "").lower() == "yes"
            )

            # Look for error bar indicators
            error_indicators = [
                "±",
                "sem",
                "standard error",
                "error bar",
                "sd",
                "standard deviation",
            ]
            has_errors_in_caption = any(
                indicator.lower() in caption.lower() for indicator in error_indicators
            )

            if has_errors_in_caption and not error_bar_on_figure:
                relevance["alignment"] = "questionable"
                relevance["notes"].append(
                    "Caption mentions error measures but model says no error bars on figure"
                )

        elif test_name == "replicates_defined":
            involves_replicates = (
                model_output.get("involves_replicates", "").lower() == "yes"
            )

            # Look for replicate indicators
            replicate_indicators = [
                "n=",
                "replicate",
                "repeat",
                "independent",
                "experiment",
            ]
            has_replicates_in_caption = any(
                indicator.lower() in caption.lower()
                for indicator in replicate_indicators
            )

            if has_replicates_in_caption and not involves_replicates:
                relevance["alignment"] = "misaligned"
                relevance["notes"].append(
                    "Caption mentions replicates but model says none involved"
                )

        return relevance

    def _check_test_issues(
        self,
        test_name: str,
        model_output: Dict[str, Any],
        caption: str,
        panel_label: str,
    ) -> Dict[str, List[str]]:
        """Check for specific issues in test results."""
        issues = {"warnings": [], "critical": []}

        # Check for inconsistencies
        if test_name == "stat_test":
            statistical_test_mentioned = model_output.get(
                "statistical_test_mentioned", ""
            )
            statistical_test_needed = model_output.get(
                "statistical_test_needed", ""
            ).lower()

            if statistical_test_needed == "yes" and statistical_test_mentioned in [
                "not mentioned",
                "",
            ]:
                issues["critical"].append(
                    f"Panel {panel_label}: Statistical test needed but not mentioned in caption"
                )

        elif test_name == "plot_axis_units":
            is_plot = model_output.get("is_a_plot", "").lower() == "yes"
            units_provided = model_output.get("units_provided", [])

            if is_plot and not units_provided:
                issues["warnings"].append(
                    f"Panel {panel_label}: Plot detected but no axis units provided"
                )

        elif test_name == "micrograph_scale_bar":
            is_micrograph = model_output.get("micrograph", "").lower() == "yes"
            scale_bar_on_image = model_output.get("scale_bar_on_image", "").lower()

            if is_micrograph and scale_bar_on_image != "yes":
                issues["warnings"].append(
                    f"Panel {panel_label}: Micrograph detected but no scale bar found"
                )

        return issues

    def _check_figure_level_issues(
        self, analysis: Dict[str, Any], caption: str
    ) -> None:
        """Check for figure-level issues."""
        panels_detected = analysis["panels_detected"]

        # Check if panel count seems reasonable
        if panels_detected == 0:
            analysis["critical_issues"].append("No panels detected in figure")
        elif panels_detected > 15:
            analysis["issues"].append(
                f"Very high panel count ({panels_detected}) - may indicate over-segmentation"
            )

        # Check for missing common elements
        panel_labels = list(analysis["panel_analysis"].keys())

        # Check for sequential panel labeling
        expected_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][
            :panels_detected
        ]
        missing_labels = [
            label for label in expected_labels if label not in panel_labels
        ]

        if missing_labels and panels_detected > 1:
            analysis["issues"].append(
                f"Missing expected panel labels: {missing_labels}"
            )

    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on the analysis."""
        recommendations = []

        total_issues = analysis_results["summary"]["issues_found"]
        critical_issues = analysis_results["summary"]["critical_issues"]

        if critical_issues > 0:
            recommendations.append(
                f"🚨 CRITICAL: Found {critical_issues} critical issues that need immediate attention"
            )

        if total_issues > 10:
            recommendations.append(
                "📊 Consider reviewing prompt engineering - high issue count suggests systematic problems"
            )

        # Analyze common issue patterns
        issue_types = {}
        for issue in analysis_results["issues"]:
            issue_text = issue["issue"]
            for key_phrase in [
                "Statistical test",
                "axis units",
                "scale bar",
                "error bars",
            ]:
                if key_phrase.lower() in issue_text.lower():
                    issue_types[key_phrase] = issue_types.get(key_phrase, 0) + 1

        for issue_type, count in issue_types.items():
            if count > 2:
                recommendations.append(
                    f"🔧 {issue_type} issues appear {count} times - consider improving related prompts"
                )

        if not recommendations:
            recommendations.append("✅ QC analysis looks good overall!")

        return recommendations

    def _save_detailed_report(
        self, analysis_results: Dict[str, Any], output_file: str
    ) -> None:
        """Save a detailed HTML report."""
        html_content = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "    <title>QC Analysis Report</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        .summary { background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }",
            "        .figure { border: 1px solid #ddd; margin: 20px 0; padding: 20px; }",
            "        .critical { color: #d32f2f; font-weight: bold; }",
            "        .warning { color: #f57c00; }",
            "        .success { color: #388e3c; }",
            "        .issue-list { margin: 10px 0; }",
            "        .recommendation { background: #e3f2fd; padding: 10px; margin: 10px 0; border-radius: 3px; }",
            "    </style>",
            "</head>",
            "<body>",
            "    <h1>QC Analysis Report</h1>",
        ]

        # Summary section
        summary = analysis_results["summary"]
        html_content.extend(
            [
                '    <div class="summary">',
                "        <h2>Summary</h2>",
                f'        <p><strong>Total Figures:</strong> {summary["total_figures"]}</p>',
                f'        <p><strong>Figures Analyzed:</strong> {summary["figures_analyzed"]}</p>',
                f'        <p><strong>Total Panels Detected:</strong> {summary["total_panels_detected"]}</p>',
                f'        <p><strong>Issues Found:</strong> <span class="warning">{summary["issues_found"]}</span></p>',
                f'        <p><strong>Critical Issues:</strong> <span class="critical">{summary["critical_issues"]}</span></p>',
                "    </div>",
            ]
        )

        # Recommendations
        html_content.append("    <h2>Recommendations</h2>")
        for rec in analysis_results["recommendations"]:
            html_content.append(f'    <div class="recommendation">{rec}</div>')

        # Issues by figure
        html_content.append("    <h2>Issues by Figure</h2>")
        for issue in analysis_results["issues"]:
            severity_class = (
                "critical" if issue["severity"] == "critical" else "warning"
            )
            html_content.append(
                f'    <div class="issue-list"><span class="{severity_class}">[{issue["severity"].upper()}]</span> '
                f'{issue["figure"]}: {issue["issue"]}</div>'
            )

        html_content.extend(["</body>", "</html>"])

        with open(output_file, "w") as f:
            f.write("\n".join(html_content))

        logger.info(f"Detailed report saved to: {output_file}")

    def print_summary(self, analysis_results: Dict[str, Any]) -> None:
        """Print a summary of the analysis results."""
        print("\n" + "=" * 60)
        print("🔍 QC ANALYSIS RESULTS")
        print("=" * 60)

        summary = analysis_results["summary"]
        print(f"📊 Total Figures: {summary['total_figures']}")
        print(f"🔬 Figures Analyzed: {summary['figures_analyzed']}")
        print(f"📋 Total Panels Detected: {summary['total_panels_detected']}")
        print(f"⚠️  Issues Found: {summary['issues_found']}")
        print(f"🚨 Critical Issues: {summary['critical_issues']}")

        print("\n📝 RECOMMENDATIONS:")
        for i, rec in enumerate(analysis_results["recommendations"], 1):
            print(f"{i}. {rec}")

        # Show top issues by figure
        if analysis_results["issues"]:
            print("\n🎯 TOP ISSUES:")
            for i, issue in enumerate(analysis_results["issues"][:5], 1):
                severity_icon = "🚨" if issue["severity"] == "critical" else "⚠️"
                print(f"{i}. {severity_icon} {issue['figure']}: {issue['issue']}")

            if len(analysis_results["issues"]) > 5:
                print(f"   ... and {len(analysis_results['issues']) - 5} more issues")


def main() -> None:
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Analyze QC results against actual figure content"
    )
    parser.add_argument("qc_results", help="Path to QC results JSON file")
    parser.add_argument("figure_data", help="Path to figure data JSON file")
    parser.add_argument("--report", "-r", help="Path to save detailed HTML report")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    analyzer = QCAnalysis()
    results = analyzer.analyze_qc_results(
        args.qc_results, args.figure_data, args.report
    )

    analyzer.print_summary(results)


if __name__ == "__main__":
    main()
