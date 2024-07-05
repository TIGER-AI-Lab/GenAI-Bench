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
- [ ] add blip2, instructblip, cogvlm, idefics2, llava, llavanext, minicpm, phi3-vision, vila, gpt4o, gemini results to leaderboard
- [ ] phi3-vision is not supported yet, need to add it to the mllm tools.


## Current Leaderboard 
(Updated on 2024-07-05, copied from [`genaibench_results.txt`](./genaibench_results.txt))

+------------+----------+---------------------------+------------------------+---------------------------+
|   Model    | Template | Image Generation Accuracy | Image Editing Accuracy | Video Generation Accuracy |
+------------+----------+---------------------------+------------------------+---------------------------+
|  kosmos2   | pairwise |            0.0            |          0.0           |            0.0            |
|  idefics2  | pairwise |          42.2478          |        27.3123         |            TBD            |
|   cogvlm   | pairwise |            TBD            |          0.0           |            TBD            |
| otterimage | pairwise |            0.0            |          0.0           |            0.0            |
| ottervideo | pairwise |            0.0            |          0.0           |            0.0            |
|   blip2    | pairwise |          26.3401          |        26.0065         |          16.9317          |
|  idefics1  | pairwise |           0.8069          |         5.6583         |           0.1871          |
|   qwenVL   | pairwise |          26.6282          |        14.9075         |            TBD            |
|   random   | pairwise |          25.3602          |        25.8977         |          25.1637          |
+------------+----------+---------------------------+------------------------+---------------------------+








## Citation
```bibtex
@article{jiang2024genai,
  title={GenAI Arena: An Open Evaluation Platform for Generative Models},
  author={Jiang, Dongfu and Ku, Max and Li, Tianle and Ni, Yuansheng and Sun, Shizhuo and Fan, Rongqi and Chen, Wenhu},
  journal={arXiv preprint arXiv:2406.04485},
  year={2024}
}
```