# GenAI-Bench
## Installation
```bash
pip install -e .
```

## Usage

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


## TODO
- [ ] add phi3-vision, vila results to leaderboard
- [ ] phi3-vision is not supported yet, need to add it to the mllm tools.
- [ ] add claude



## Current Leaderboard 
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







## Citation
```bibtex
@article{jiang2024genai,
  title={GenAI Arena: An Open Evaluation Platform for Generative Models},
  author={Jiang, Dongfu and Ku, Max and Li, Tianle and Ni, Yuansheng and Sun, Shizhuo and Fan, Rongqi and Chen, Wenhu},
  journal={arXiv preprint arXiv:2406.04485},
  year={2024}
}
```