[![arXiv](https://img.shields.io/badge/arXiv-2309.00236-b31b1b.svg)](https://arxiv.org/abs/2309.00236)

# Image Hijacks: Adversarial Images can Control Generative Models at Runtime

This is the code for _Image Hijacks: Adversarial Images can Control Generative Models at Runtime_.

- [Project page and demo](https://image-hijacks.github.io)
- [Paper](https://arxiv.org/abs/2309.00236)

## Setup

The code can be run under any environment with Python 3.9 and above. 

We use [poetry](https://python-poetry.org) for dependency management, which can be installed following the instructions [here](https://python-poetry.org/docs/#installation).

To build a virtual environment with the required packages, simply run

```bash
poetry install
```

Notes
- On some systems you may need to set the environment variable `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` to avoid keyring-based errors.
- This codebase stores large files (e.g. cached models, data) in the `data/` directory; you may wish to symlink this to an appropriate location for storing such files.

## Training

The images used in our [demo](https://image-hijacks.github.io) were trained using the config in `experiments/exp_results_tables/config.py` (specifically runs #1 `llava1_att_leak.pat_full.eps_8.lr_3e-2` and #5 `llava1_att_spec.pat_full.eps_8.lr_3e-2`).

To train these images, first download the relevant LLaVA checkpoint:

```bash
poetry run python download.py models llava-v1.3-13b-336px
```

To get the list of jobs (with their job IDs) specified by this config file:

```bash
poetry run python experiments/exp_demo_imgs/config.py
```

To run job ID `N` without [wandb](https://wandb.ai/) logging:

```bash
poetry run python run.py train \
--config_path experiments/exp_demo_imgs/config.py \
--log_dir experiments/exp_demo_imgs/logs \
--job_id N \
--playground
```

To run job ID `N` with [wandb](https://wandb.ai/) logging to `YOUR_WANDB_ENTITY/YOUR_WANDB_PROJECT`:

```bash
poetry run python run.py train \
--config_path experiments/exp_results_tables/config.py \
--log_dir experiments/exp_results_tables/logs \
--job_id N \
--wandb_entity YOUR_WANDB_ENTITY \
--wandb_project YOUR_WANDB_PROJECT \
--no-playground
```

Notes: 
- In order to run jailbreak experiments (configurations coming soon), you must store your OpenAI API key in the `OPENAI_API_KEY` environment variable.

## Tests

This codebase advocates for [expect tests](https://blog.janestreet.com/the-joy-of-expect-tests) in machine learning, and as such uses @ezyang's [expecttest](https://github.com/ezyang/expecttest) library for unit and regression tests.

To run tests,

```bash
poetry run python download.py models blip2-flan-t5-xl
poetry run pytest .
```

## Citation

To cite our work, you can use the following BibTeX entry:

```bibtex
@misc{bailey2023image,
  title={Image Hijacks: Adversarial Images can Control Generative Models at Runtime}, 
  author={Luke Bailey and Euan Ong and Stuart Russell and Scott Emmons},
  year={2023},
  eprint={2309.00236},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
