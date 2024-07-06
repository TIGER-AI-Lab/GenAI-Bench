import fire
import json
import prettytable as pt
import numpy as np
from pathlib import Path
def load_results(results_dir):
    all_results = {}
    for task in results_dir.iterdir():
        all_results[task.name] = []
        for model in task.iterdir():
            for template in model.glob("*.json"):
                with open(template) as f:
                    single_model_results = json.load(f)
                all_results[task.name].append({
                    "model": model.name,
                    "template": template.stem,
                    "raw_data": single_model_results,
                    "acc": np.mean([x["correct"] for x in single_model_results])
                })
                
    # sort the results by accuracy
    for task in all_results:
        all_results[task] = sorted(all_results[task], key=lambda x: x["acc"], reverse=True)
    return all_results

def main(
    results_dir="./results",
    save_file="genaibench_results.txt"
):
    results_dir = Path(results_dir)
    all_results = load_results(results_dir)
    
    
    f_save_file = open(save_file, "w")
    for task in all_results:
        table = pt.PrettyTable()
        table.field_names = ["Model", "Template", "Accuracy"]
        for result in all_results[task]:
            table.add_row([result["model"], result["template"], round(result["acc"]*100, 4)])
        table.set_style(pt.DEFAULT)
        task_str = task.replace("_", " ").title()
        print("### " + task_str)
        print(table)
        print("\n\n")
        table.set_style(pt.MARKDOWN)
        print("### " + task_str, file=f_save_file)
        print(table, file=f_save_file)
        print("\n\n", file=f_save_file)
    f_save_file.close()
        
    
def main_merged(results_dir="./results", save_file="genaibench_results.txt"):
    results_dir = Path(results_dir)
    all_results = load_results(results_dir)
    
    f_save_file = open(save_file, "w")
    
    # Initialize a dictionary to store the merged results
    merged_results = {}

    # Collect all unique models and templates
    all_models = set()
    all_templates = set()
    for task in all_results:
        for result in all_results[task]:
            all_models.add(result["model"])
            all_templates.add(result["template"])
    
    # Initialize the merged_results dictionary with models and templates
    for model in all_models:
        merged_results[model] = {}
        for template in all_templates:
            merged_results[model][template] = {"image_generation": "TBD", "image_edition": "TBD", "video_generation": "TBD"}
    
    # Fill the merged_results dictionary with actual results
    for task in all_results:
        for result in all_results[task]:
            merged_results[result["model"]][result["template"]][task] = round(result["acc"] * 100, 4)
    
    # Sort models, putting 'random' at the top
    sorted_models = sorted(all_models, key=lambda x: (x != 'random', x))
    
    # Create the merged table
    table = pt.PrettyTable()
    table.field_names = ["Model", "Template", "Image Generation Accuracy", "Image Editing Accuracy", "Video Generation Accuracy"]
    
    for model in sorted_models:
        for template in merged_results[model]:
            table.add_row([
                model,
                template,
                merged_results[model][template]["image_generation"],
                merged_results[model][template]["image_edition"],
                merged_results[model][template]["video_generation"],
            ])
    
    table.set_style(pt.DEFAULT)
    print("### Merged Results")
    print(table)
    print("\n\n")
    table.set_style(pt.MARKDOWN)
    print("### Merged Results", file=f_save_file)
    print(table, file=f_save_file)
    print("\n\n", file=f_save_file)
    
    f_save_file.close()

if __name__ == "__main__":
    fire.Fire(main_merged)