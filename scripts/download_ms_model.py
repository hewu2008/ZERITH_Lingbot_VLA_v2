import argparse
import os

from modelscope import snapshot_download


"""
python3 scripts/download_ms_model.py --repo_id Robbyant/lingbot-vla-v2-6b --local_dir ./lingbot-vla-v2-6b
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="Robbyant/lingbot-vla-v2-6b")
    parser.add_argument("--local_dir", type=str, default="./lingbot-vla-v2-6b")
    args = parser.parse_args()

    repo_id = args.repo_id
    local_dir = args.local_dir

    try:
        snapshot_download(
            model_id=repo_id,
            local_dir=os.path.join(local_dir, repo_id.split("/")[1]),
        )
    except TypeError:
        snapshot_download(
            model_id=repo_id,
            cache_dir=local_dir,
        )