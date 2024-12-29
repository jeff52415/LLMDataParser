import secrets
from functools import lru_cache

import gradio as gr

from llmdataparser import ParserRegistry
from llmdataparser.base_parser import DatasetDescription, EvaluationMetric, ParseEntry


@lru_cache(maxsize=32)
def get_parser_instance(parser_name: str):
    """Get a cached parser instance by name."""
    return ParserRegistry.get_parser(parser_name)


def get_available_splits(parser) -> list[str] | None:
    """Get available splits for the selected parser after loading."""
    if not hasattr(parser, "split_names") or not parser.split_names:
        return None
    return parser.split_names


def get_available_tasks(parser) -> list[str]:
    """Get available tasks for the selected parser."""
    if not hasattr(parser, "task_names"):
        return ["default"]
    return parser.task_names


def format_entry_attributes(entry: ParseEntry) -> str:
    """Format all attributes of a ParseEntry except prompt and answer."""
    from dataclasses import fields

    # Get all field names from the dataclass
    field_names = [field.name for field in fields(entry)]
    # Filter out prompt and answer
    filtered_fields = [name for name in field_names if name not in ["prompt", "answer"]]
    # Build the formatted string
    return "\n".join(f"{name}: {getattr(entry, name)}" for name in filtered_fields)


def load_and_parse(
    parser_name: str, task_name: str | None, split_name: str | None
) -> tuple:
    """Load and parse the dataset, return the first entry and available splits."""
    try:
        parser = get_parser_instance(parser_name)

        # Load the dataset
        parser.load(
            task_name=task_name if task_name != "default" else None,
            split=split_name,
            trust_remote_code=True,
        )

        # Get available splits after loading
        available_splits = get_available_splits(parser)

        # Parse the dataset
        parser.parse(split_names=split_name, force=True)

        # Get parsed data
        parsed_data = parser.get_parsed_data

        split_dropdown = gr.Dropdown(
            choices=available_splits,
            label="Select Split",
            interactive=True,
            value=None,
            allow_custom_value=True,
        )

        info = parser.__repr__()
        if not parsed_data:
            return 0, "No entries found", "", "", split_dropdown, info

        # Get the first entry
        first_entry = parsed_data[0]

        return (
            0,  # Return first index instead of list of indices
            first_entry.prompt,
            first_entry.raw_question,
            first_entry.answer,
            format_entry_attributes(first_entry),
            split_dropdown,
            info,
        )
    except Exception as e:
        # Make the error message more user-friendly and detailed
        error_msg = f"Failed to load dataset: {str(e)}\nParser: {parser_name}\nTask: {task_name}\nSplit: {split_name}"
        return 0, error_msg, "", "", "", [], ""


def update_entry(parsed_data_index: int | None, parser_name: str):
    """Update the displayed entry based on the selected index."""
    try:
        if not parser_name:
            return "Please select a parser first", "", "", ""

        parser = get_parser_instance(parser_name)
        parsed_data = parser.get_parsed_data

        if not parsed_data:
            return "No data available", "", "", ""

        if parsed_data_index is None:
            # Random selection using secrets instead of random
            random_index = secrets.randbelow(len(parsed_data))
            entry = parsed_data[random_index]
        else:
            # Ensure index is within bounds
            index = max(0, min(parsed_data_index, len(parsed_data) - 1))
            entry = parsed_data[index]

        return (
            entry.prompt,
            entry.raw_question,
            entry.answer,
            format_entry_attributes(entry),
        )
    except Exception as e:
        return f"Error: {str(e)}", "", ""


def update_parser_options(parser_name: str) -> tuple[gr.Dropdown, gr.Dropdown, str]:
    """Update available tasks and splits for the selected parser."""
    try:
        parser = get_parser_instance(parser_name)
        tasks = get_available_tasks(parser)
        default_task = getattr(parser, "_default_task", "default")

        # Update task dropdown
        task_dropdown = gr.Dropdown(
            choices=tasks,
            value=default_task,
            label="Select Task",
            interactive=True,
            allow_custom_value=True,
        )

        # Update split dropdown - Note the value is now explicitly None
        splits = get_available_splits(parser)
        split_dropdown = gr.Dropdown(
            choices=splits,
            label="Select Split",
            interactive=True,
            value=None,
            allow_custom_value=True,
        )

        info = parser.__repr__()
        return task_dropdown, split_dropdown, info
    except Exception as e:
        return (
            gr.Dropdown(choices=["default"], value="default"),
            gr.Dropdown(choices=[]),
            f"Error: {str(e)}",
        )


def clear_parser_cache():
    """Clear the parser cache."""
    get_parser_instance.cache_clear()


def format_dataset_description(description: DatasetDescription) -> str:
    """Format DatasetDescription into a readable string."""
    formatted = [
        f"# {description.name}",
        f"\n**Purpose**: {description.purpose}",
        f"\n**Language**: {description.language}",
        f"\n**Format**: {description.format}",
        f"\n**Source**: {description.source}",
        f"\n**Characteristics**: {description.characteristics}",
    ]

    if description.citation:
        formatted.append(f"\n**Citation**:\n```\n{description.citation}\n```")

    if description.additional_info:
        formatted.append("\n**Additional Information**:")
        for key, value in description.additional_info.items():
            formatted.append(f"- {key}: {value}")

    return "\n".join(formatted)


def get_primary_metrics(metrics: list[EvaluationMetric]) -> list[str]:
    """Get list of primary metric names."""
    return [metric.name for metric in metrics if metric.primary]


def format_metric_details(metric: EvaluationMetric) -> str:
    """Format a single EvaluationMetric into a readable string."""
    return f"""# {metric.name}<br>
                **Type**: {metric.type}<br>
                **Description**: {metric.description}"""


def update_dataset_info(parser_name: str) -> tuple:
    """Update dataset description and evaluation metrics information."""
    try:
        parser = get_parser_instance(parser_name)
        description = parser.get_dataset_description()
        metrics = parser.get_evaluation_metrics()

        # Format description
        desc_text = format_dataset_description(description)

        # Get primary metrics for dropdown
        primary_metrics = get_primary_metrics(metrics)

        # Format details for first metric (or empty if no metrics)
        first_metric = metrics[0] if metrics else None
        metric_details = format_metric_details(first_metric) if first_metric else ""

        return (
            gr.Markdown(value=desc_text),
            gr.Dropdown(
                choices=primary_metrics,
                value=primary_metrics[0] if primary_metrics else None,
            ),
            gr.Markdown(value=metric_details),
        )
    except Exception as e:
        return (
            gr.Markdown(value=f"Error loading dataset description: {str(e)}"),
            gr.Dropdown(choices=[]),
            gr.Markdown(value=""),
        )


def update_metric_details(metric_name: str, parser_name: str) -> str:
    """Update the displayed metric details when selection changes."""
    try:
        parser = get_parser_instance(parser_name)
        metrics = parser.get_evaluation_metrics()
        selected_metric = next((m for m in metrics if m.name == metric_name), None)
        return format_metric_details(selected_metric) if selected_metric else ""
    except Exception as e:
        return f"Error loading metric details: {str(e)}"


def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# LLM Evaluation Dataset Parser")

        # State management
        parser_state = gr.State("")
        dataset_status = gr.Textbox(label="Dataset Status", interactive=False)

        with gr.Tabs():
            with gr.Tab("Dataset Explorer"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Parser selection and controls
                        available_parsers = ParserRegistry.list_parsers()
                        parser_dropdown = gr.Dropdown(
                            choices=available_parsers,
                            label="Select Parser",
                            value=available_parsers[0] if available_parsers else None,
                            interactive=True,
                            allow_custom_value=True,
                        )
                        task_dropdown = gr.Dropdown(
                            choices=["default"],
                            label="Select Task",
                            value="default",
                            interactive=True,
                            allow_custom_value=True,
                        )
                        split_dropdown = gr.Dropdown(
                            choices=[],
                            label="Select Split",
                            interactive=True,
                            value=None,
                            allow_custom_value=True,
                        )
                        load_button = gr.Button(
                            "Load and Parse Dataset", variant="primary"
                        )

                        # Entry selection
                        entry_index = gr.Number(
                            label="Select Entry Index (empty for random)",
                            precision=0,
                            interactive=True,
                        )
                        update_button = gr.Button(
                            "Update/Random Entry", variant="secondary"
                        )

                    with gr.Column(scale=2):
                        # Output displays
                        prompt_output = gr.Textbox(
                            label="Prompt", lines=5, show_copy_button=True
                        )
                        raw_question_output = gr.Textbox(
                            label="Raw Question", lines=5, show_copy_button=True
                        )
                        answer_output = gr.Textbox(
                            label="Answer", lines=5, show_copy_button=True
                        )
                        attributes_output = gr.Textbox(
                            label="Other Attributes", lines=5, show_copy_button=True
                        )

            with gr.Tab("Dataset Information"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Dataset description
                        dataset_description = gr.Markdown()

                    with gr.Column(scale=1):
                        # Evaluation metrics
                        gr.Markdown("## Evaluation Metrics")
                        metric_dropdown = gr.Dropdown(
                            label="Select Primary Metric", interactive=True
                        )
                        metric_details = gr.Markdown()

        # Event handlers
        parser_dropdown.change(
            fn=update_parser_options,
            inputs=parser_dropdown,
            outputs=[
                task_dropdown,
                split_dropdown,
                dataset_status,
            ],
        ).then(lambda x: x, inputs=parser_dropdown, outputs=parser_state).then(
            fn=update_dataset_info,
            inputs=[parser_dropdown],
            outputs=[dataset_description, metric_dropdown, metric_details],
        )

        load_button.click(
            fn=load_and_parse,
            inputs=[parser_dropdown, task_dropdown, split_dropdown],
            outputs=[
                entry_index,
                prompt_output,
                raw_question_output,
                answer_output,
                attributes_output,
                split_dropdown,
                dataset_status,
            ],
            api_name="load_and_parse",
            show_progress="full",
        ).then(
            fn=update_dataset_info,
            inputs=[parser_dropdown],
            outputs=[dataset_description, metric_dropdown, metric_details],
        )

        update_button.click(
            fn=update_entry,
            inputs=[entry_index, parser_state],
            outputs=[
                prompt_output,
                raw_question_output,
                answer_output,
                attributes_output,
            ],
            api_name="update_entry",
        )

        metric_dropdown.change(
            fn=update_metric_details,
            inputs=[metric_dropdown, parser_dropdown],
            outputs=metric_details,
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False)  # Enable sharing for remote access
