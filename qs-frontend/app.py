import streamlit as st
import json
import argparse
from clearml.backend_api.session.client import APIClient

st.set_page_config(page_title="Model development interface")


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config


def create_dropdowns(dropdowns):
    selections = {}
    for dropdown in dropdowns:
        label = dropdown.get("label", "Select an option")
        options = dropdown.get("options", [])
        selection = st.selectbox(label, options)
        selections[label] = selection

    return selections


def run_pipeline(selections, task_id: str):
    selections = list(selections.values())

    client = APIClient()
    clone_of_pipeline = client.tasks.clone(
        new_project_name="Thesis-API-Test",
        task=task_id,
        new_task_hyperparams={
            "Args": {
                "collection": {"section": "Args", "name": "collection", "value": selections[1]},
                "connection_string": {"section": "Args", "name": "connection_string", "value": selections[4]},
                "database": {"section": "Args", "name": "database", "value": selections[0]},
                "id": {"section": "Args", "name": "id", "value": selections[2]}
            }
        })
    queued_task = client.tasks.enqueue(task=clone_of_pipeline.id, queue_name=selections[3])

    print(clone_of_pipeline)
    print(clone_of_pipeline.id)
    print("----------")
    print(queued_task)
    print("-------------------------------")

    return clone_of_pipeline.id, clone_of_pipeline.new_project


def main(config_file_path: str, task_id: str):
    st.title("Model development interface")
    config = load_config(config_file_path)

    if config and "dropdowns" in config:
        selections = create_dropdowns(config["dropdowns"])
        if st.button("Start the pipeline"):
            task_id, new_project = run_pipeline(selections, task_id)
            st.text("-----------------------------------------------------")
            st.subheader("Task details")
            st.text(f"Task ID: {task_id}")
            st.text(f"Project ID: {new_project}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QS-APP with Configurable Dropdowns")
    parser.add_argument(
        '--config-file',
        type=str,
        help="Path to the JSON configuration file."
    )
    parser.add_argument(
        '--taskid',
        type=str,
        help="The task id of the pipeline you want to use as a template."
    )
    args = parser.parse_args()

    main(args.config_file, args.taskid)
