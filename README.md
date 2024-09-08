# GenAI-Bench
<a target="_blank" href="https://arxiv.org/abs/2406.04485">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena">
<img style="height:22pt" src="https://img.shields.io/badge/-🌐%20GenAI_Arena-blue?style=flat"></a>
<a target="_blank" href="https://huggingface.co/datasets/TIGER-Lab/GenAI-Bench">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20GenAI_Bench-red?style=flat"></a>
<!-- <a target="_blank" href="https://huggingface.co/spaces/TIGER-Lab/Mantis"> -->
<!-- <img style="height:22pt" src="https://img.shields.io/badge/-Tweet-blue?style=flat&logo=twitter"></a> -->
<br>

---

## Introduction
GenAI-Bench is a benchmark designed to benchmark MLLMs’s ability in judging the quality of AI generative contents by comparing with human preferences collected through our [🤗 GenAI-Arnea](https://huggingface.co/spaces/TIGER-Lab/GenAI-Arena). In other words, we are evaluting the capabilities of existing MLLMs as a multimodal reward model, and in this view, GenAI-Bench is a reward-bench for multimodal generative models.

We filter existing votes collecte visa NSFW filter and other heuristics, and then finally resulting in 1735 votes for image generation, 919 votes for image editing, and 1069 votes for video generation, which is used to evaluate the performance of MLLMs on aligning with human preferences. 

We adopts a pairwise comparison template for each tasks, where the model is asked to output 4 labels for each pair of AI generative contents, which are `A>B`, `B>A`, `A=B=Good`, `A=B=Bad`. We then calculate the average accuracy of the model by comparing the model's prediction with the human preference. The prompt templates are shown below:
| Task | Template File |
| :---: | :---: |
| Image Generation | [./templates/image_generation/pairwise.txt](./templates/image_generation/pairwise.txt) |
| Image Editing | [./templates/image_edition/pairwise.txt](./templates/image_edition/pairwise.txt) |
| Video Generation | [./templates/video_generation/pairwise.txt](./templates/video_generation/pairwise.txt) |

The leaderboard is updated every time a new model is evaluated.
## Installation
```bash
pip install -e .
```

## Evaluate a model

- run inference of a model on a task
```bash
python inference.py --task "video_generation" --model_name "random"
python inference.py --task "video_generation" --model_name "gpt4o"
```

or run it through `inference.sh`:
```bash
./inference.sh <GPU_ID> <MODEL_NAME> [<TASK_ID>] # 0 is image generation, 1 is image edition, 2 is video generation 
```

- The default prompt template is `pairwise` for each tasks ([`image_generation`](./genaibench/templates/image_generation/pairwise.txt), [`image_edition`](./genaibench/templates/image_edition/pairwise.txt), [`video_generation`](./genaibench/templates/video_generation/pairwise.txt)), you can write your own prompt template and pass it to the `--template` argument.

- Show existing results of the leaderboard
```bash
python show_results.py
```
Then results will be printed and saveed to [`genaibench_results.txt`](./genaibench_results.txt).


## Contributing a new model
If you want to evaluate your model on GenAI-Bench, you can follow the steps below:
1. Fork this repository
2. Follow [./genaibench/mllm_tools/README.md](./genaibench/mllm_tools/README.md) to add your model to the evaluation pipeline.
3. Run the evaluation script and update the results in the leaderboard in the README.
4. Create a pull request to this repository.

## Current Leaderboard (v1 split)
(Updated on 2024-08-09, copied from [`genaibench_results.txt`](./genaibench_results.txt))
|          Model          | Template | Image Generation | Image Editing | Video Generation | Average |
| :---------------------: | :------: | :--------------: | :-----------: | :--------------: | :-----: |
|          random         | pairwise |      25.36       |      25.9     |      25.16       |  25.47  |
|          gpt4o          | pairwise |      45.59       |     53.54     |      48.46       |   49.2  |
|      gemini-1.5-pro     | pairwise |      44.67       |     55.93     |      46.21       |  48.94  |
|          llava          | pairwise |       37.0       |     26.12     |       30.4       |  31.17  |
|         idefics2        | pairwise |      42.25       |     27.31     |      16.46       |  28.67  |
|        llavanext        | pairwise |      22.65       |     25.35     |       21.7       |  23.23  |
|      minicpm-V-2.5      | pairwise |      37.81       |     25.24     |       6.55       |   23.2  |
|          blip2          | pairwise |      26.34       |     26.01     |      16.93       |  23.09  |
|        videollava       | pairwise |      37.75       |     26.66     |       0.0        |  21.47  |
|          cogvlm         | pairwise |      29.34       |      0.0      |       24.6       |  17.98  |
|          qwenVL         | pairwise |      26.63       |     14.91     |       2.15       |  14.56  |
|       instructblip      | pairwise |       3.11       |      19.8     |       3.74       |   8.88  |
|         idefics1        | pairwise |       0.81       |      5.66     |       0.19       |   2.22  |
|        ottervideo       | pairwise |       0.0        |      0.0      |       0.0        |   0.0   |
|        otterimage       | pairwise |       0.0        |      0.0      |       0.0        |   0.0   |
|         kosmos2         | pairwise |       0.0        |      0.0      |       0.0        |   0.0   |




## TODO
We are planning to add more models to the leaderboard, and the following are the tasks that need to be done. We welcome contributions from the community and your help will be greatly appreciated.
- [ ] Phi-3-vision
- [ ] InternVL
- [ ] Phi3-vision
- [ ] VILA.
- [ ] Claude

## Citation
```bibtex
@article{jiang2024genai,
  title={GenAI Arena: An Open Evaluation Platform for Generative Models},
  author={Jiang, Dongfu and Ku, Max and Li, Tianle and Ni, Yuansheng and Sun, Shizhuo and Fan, Rongqi and Chen, Wenhu},
  journal={arXiv preprint arXiv:2406.04485},
  year={2024}
}
```
