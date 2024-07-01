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
        
    

if __name__ == "__main__":
    fire.Fire(main)