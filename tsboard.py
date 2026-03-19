#!/usr/bin/env python3
# coding: utf-8

import os , argparse , subprocess
from src.proj import PATH , Logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    os.chdir(PATH.main)
    model_name = parse_args().model
    if model_name is None:
        from src.api.model import ModelAPI
        from src.res.model.util.model_path import ModelPath
        candidates = [ModelPath(model) for model in ModelAPI.available_models(include_short_test = True)]
        candidates = [model for model in candidates if model.snapshot('tensorboard').exists()]
        candidates_fit_time = [model.log_file.read_entry(max_entries = 1 , pattern = 'fit_model >> end') for model in candidates]
        Logger.success("Choose from available models that have tensorboard logs when no model name is provided:")
        for i , (model , fit_time_entries) in enumerate(zip(candidates, candidates_fit_time)):
            model_info = f"{model.full_name} - Fit Time: {fit_time_entries[0].timestamp if fit_time_entries else 'N/A'}"
            Logger.stdout(f"{i + 1:>2}. {model_info}" , indent = 1 , color = 'lightyellow')
        index = int(input("Enter the number of the model to launch: "))
        assert index >= 1 and index <= len(candidates) , f"Invalid index: {index} , must be between 1 and {len(candidates)}"
        model_name = candidates[index - 1].full_name
    model_path = ModelPath(model_name)
    # cmd = ["uv", "run", "tensorboard", "--logdir", model_path.snapshot('tensorboard').as_posix()]

    try:
        # This will keep the process running until you hit Ctrl+C
        # subprocess.run(cmd, check=True)
        os.system(f"uv run tensorboard --logdir {model_path.snapshot('tensorboard').as_posix()}")
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to launch: {e}")
