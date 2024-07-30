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
- [ ] Gemini-1.5-flash
- [ ] GPT4o-mini



## Current Leaderboard 
(Updated on 2024-07-29, copied from [`genaibench_results.txt`](./genaibench_results.txt))

|          Model          | Template | Image Generation Accuracy | Image Editing Accuracy | Video Generation Accuracy |
| :---------------------: | :------: | :-----------------------: | :--------------------: | :-----------------------: |
|          random         | pairwise |          25.3602          |        25.8977         |          25.1637          |
|          blip2          | pairwise |          26.3401          |        26.0065         |          16.9317          |
|          cogvlm         | pairwise |          29.3372          |          0.0           |          24.6024          |
|      gemini-1.5-pro     | pairwise |          44.6686          |        55.9304         |          46.2114          |
|          gpt4o          | pairwise |          45.5908          |        53.5365         |          48.4565          |
|         idefics1        | pairwise |           0.8069          |         5.6583         |           0.1871          |
|         idefics2        | pairwise |          42.2478          |        27.3123         |           16.464          |
|       instructblip      | pairwise |           3.1124          |        19.8041         |           3.7418          |
|         kosmos2         | pairwise |            0.0            |          0.0           |            0.0            |
|          llava          | pairwise |          37.0029          |        26.1153         |          30.4022          |
|        llavanext        | pairwise |          22.6513          |        25.3536         |          21.7025          |
|    mantis-8b-idefics2   | pairwise |            0.0            |          0.0           |            0.0            |
| mantis-8b-idefics2_8192 | pairwise |            0.0            |          0.0           |            0.0            |
|      minicpm-V-2.5      | pairwise |          37.8098          |        25.2448         |           6.5482          |
|        otterimage       | pairwise |            0.0            |          0.0           |            0.0            |
|        ottervideo       | pairwise |            0.0            |          0.0           |            0.0            |
|          qwenVL         | pairwise |          26.6282          |        14.9075         |           2.1515          |
|        videollava       | pairwise |          37.7522          |        26.6594         |            0.0            |














## Citation
```bibtex
@article{jiang2024genai,
  title={GenAI Arena: An Open Evaluation Platform for Generative Models},
  author={Jiang, Dongfu and Ku, Max and Li, Tianle and Ni, Yuansheng and Sun, Shizhuo and Fan, Rongqi and Chen, Wenhu},
  journal={arXiv preprint arXiv:2406.04485},
  year={2024}
}
```