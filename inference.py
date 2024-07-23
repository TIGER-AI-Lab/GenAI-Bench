import fire
import datasets
import json
import numpy as np
import av
import regex as re
import requests
from PIL import Image
from tqdm import tqdm
from genaibench.mllm_tools import MLLM_Models
from genaibench.utils import (
    load_template,
    process_video_into_frames,
)
from pathlib import Path

def run_example(model, example, input_keys, input_types, prompt_template, inference_configs):
    prompt = prompt_template
    model_inputs = []
    for i, key in enumerate(input_keys):
        if key.startswith("left"):
            actual_key = key.replace("left_", "")
        elif key.startswith("right"):
            actual_key = key.replace("right_", "")
        else:
            actual_key = key
        if input_types[i] == "str":
            prompt = prompt.replace(f"<{actual_key}>", example[key])
        elif input_types[i] == "image":
            splitted_prompt = prompt.split(f"<{actual_key}>")
            assert len(splitted_prompt) == 2, f"Prompt: {prompt}, Key: {actual_key}"
            model_inputs.append({
                "type": "text",
                "content": splitted_prompt[0]
            })
            model_inputs.append({
                "type": "image",
                "content": example[key]
            })
            prompt = splitted_prompt[1]
        elif input_types[i] == "video":
            splitted_prompt = prompt.split(f"<{actual_key}>")
            assert len(splitted_prompt) == 2
            model_inputs.append({
                "type": "text",
                "content": splitted_prompt[0]
            })
            frames = process_video_into_frames(example[key], inference_configs.get("max_num_frames", 16))
            if hasattr(model, "support_video_input") and model.support_video_input:
                model_inputs.append({
                    "type": "video",
                    "content": frames
                })
            else:
                for frame in frames:
                    model_inputs.append({
                        "type": "image",
                        "content": Image.fromarray(frame).convert("RGB")
                    })
            prompt = splitted_prompt[1]
            
    model_inputs.append({
        "type": "text",
        "content": prompt
    })
    
    response = model(model_inputs)
    return response

def run_pairwise_example(model, example, left_input_keys, right_input_keys, input_types, prompt_template, inference_configs):
    prompt = prompt_template
    
    merged_keys = []
    merged_key_types = []
    for i, key in enumerate(left_input_keys):
        if key in merged_keys:
            continue
        merged_keys.append(key)
        merged_key_types.append(input_types[i])
    for i, key in enumerate(right_input_keys):
        if key in merged_keys:
            continue
        merged_keys.append(key)
        merged_key_types.append(input_types[i])
        
    multimodal_keys = {}
    
    for i, key in enumerate(merged_keys):
        if merged_key_types[i] == "str":
            prompt = prompt.replace(f"<{key}>", example[key])
        elif merged_key_types[i] == "image":
            multimodal_keys[key] = {
                "type": "image",
                "content": example[key],
                "pos": prompt.find(f"<{key}>")
            }
        elif merged_key_types[i] == "video":
            frames = process_video_into_frames(example[key], inference_configs.get("max_num_frames", 8))
            multimodal_keys[key] = {
                "type": "video",
                "content": frames,
                "path": example[key],
                "pos": prompt.find(f"<{key}>")
            }
        else:
            raise ValueError(f"Type {merged_key_types[i]} not supported.")
    
    sorted_multimodal_keys = sorted(multimodal_keys.items(), key=lambda x: x[1]["pos"])
    model_inputs = []
    for key, multimodal_ in sorted_multimodal_keys:
        sub_prompts = prompt.split(f"<{key}>")
        assert len(sub_prompts) == 2, f"Prompt: {prompt}, Key: {key}"
        model_inputs.append({
            "type": "text",
            "content": sub_prompts[0]
        })
        if multimodal_["type"] == "image":
            model_inputs.append({
                "type": "image",
                "content": multimodal_["content"]
            })
        elif multimodal_["type"] == "video":
            if hasattr(model, "support_video_input") and model.support_video_input:
                model_inputs.append({
                    "type": "video",
                    "content": multimodal_["path"]
                })
            else:
                for frame in multimodal_["content"]:
                    model_inputs.append({
                        "type": "image",
                        "content": Image.fromarray(frame).convert("RGB")
                    })
        else:
            raise ValueError(f"Type {multimodal_['type']} not supported.")
        prompt = sub_prompts[1]
    model_inputs.append({
        "type": "text",
        "content": prompt
    })
    response = model(model_inputs)
    return response

def parse_response(response, human_vote):
    """
    Parse the response and return the model's vote.
    Args:
        response (str): The response from the model. 
        human_vote (str): The human's vote. One of "leftvote", "rightvote", "tievote", "bothbad_vote"
    Returns:
        model_vote (str): parsed model's vote, one of "A=B=Bad", "A=B=Good", "A>B", "B>A"
        correct (str): accuracy of the model's vote, True if correct, False if incorrect.
    """
    model_vote = re.search(r"\[\[.*\]\]", response)
    if model_vote is None:
        return None, False
    model_vote = model_vote.group()
    correct = False
    if human_vote == "leftvote":
        if model_vote == "[[A>>B]]" or model_vote == "[[A>B]]":
            correct = True
    elif human_vote == "rightvote":
        if model_vote == "[[B>>A]]" or model_vote == "[[B>A]]":
            correct = True
    elif human_vote == "tievote":
        if model_vote == "[[A=B]]" or model_vote == "[[A=B=Good]]":
            correct = True
    elif human_vote == "bothbad_vote":
        if model_vote == "[[A=B]]" or model_vote == "[[A=B=Bad]]":
            correct = True
    return model_vote, correct
    
def main(
    task: str,
    model_name: str,
    template: str="pairwise",
    genaibench="TIGER-Lab/GenAI-Bench",
    inference_configs=Path(__file__).parent / "inference_configs.json",
    overwrite: bool=False,
    results_dir=None,
):
    assert task in ["image_generation", "image_edition", "video_generation"], f"Task {task} not supported."
    if not model_name == "random":
        model = MLLM_Models(model_name)()
    else:
        model = None
    dataset = datasets.load_dataset(genaibench, task, split='train') # the train split should be the test split we use. it's a bug in the dataset.
    
    if results_dir is None:
        results_dir = Path(__file__).parent / "results"
    else:
        results_dir = Path(results_dir)
    results_file = results_dir / task / model_name / f"{template}.jsonl"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(inference_configs) as f:
        inference_configs = json.load(f)[task]
    
    prompt_template = load_template(task, template)
    
    if results_file.exists() and not overwrite:
        with open(results_file) as f:
            existing_results = [json.loads(x) for x in f]
        if len(existing_results) == len(dataset):
            print(f"Results already exist at {results_file})")
        elif len(existing_results) < len(dataset):
            print(f"Results already exist at {results_file}. Continuing from where it left off.")
        else:
            print(f"Results file {results_file} has more results than the dataset. Overwriting.")
            results_file.unlink()
            existing_results = []
    else:
        existing_results = []
    existing_results = {x["idx"]: x for x in existing_results}
        
    
    all_outputs = []
    with open(results_file, "a+") as f:
    
        for i, example in tqdm(enumerate(dataset), desc="Running Inference", total=len(dataset)):
            
            if i in existing_results:
                outputs = existing_results[i]
                all_outputs.append(outputs)
                continue
            
            left_input_keys = inference_configs["input_keys"]["left"]
            right_input_keys = inference_configs["input_keys"]["right"]
            input_key_types = inference_configs["input_types"]
            
            if template == "pairwise":
                outputs = {"idx": i}
                if model_name != "random":
                    response = run_pairwise_example(model, example, left_input_keys, right_input_keys, input_key_types, prompt_template, inference_configs)
                else:
                    response = np.random.choice(["[[A=B=Good]]", "[[A=B=Bad]]", "[[A>B]]", "[[B>A]]"])
                model_vote, correct = parse_response(response, example["vote_type"])
                
                for i, key in enumerate(left_input_keys):
                    if input_key_types[i] == "str":
                        outputs[key] = example[key]
                outputs['response'] = response
                outputs['model_vote'] = model_vote
                outputs['human_vote'] = example['vote_type']
                outputs['correct'] = correct
                # print(f"Response: {response}")
                # print(f"Model Vote: {model_vote}")
                # print(f"Human Vote: {example['vote_type']}")
                # print(f"Correct: {correct}")
            else:
                raise ValueError(f"Template {template} not supported.")
                
            f.write(json.dumps(outputs) + "\n")
            all_outputs.append(outputs)
    
    # save as a whole in json
    json_results_file = results_file.with_suffix(".json")
    with open(json_results_file, "w") as f:
        json.dump(all_outputs, f, indent=4)
    
    print(f"Results saved to {json_results_file}")
    # print acc
    print(f"Accuracy: {np.mean([x['correct'] for x in all_outputs])}")
    
    
if __name__ == "__main__":
    fire.Fire(main)
    