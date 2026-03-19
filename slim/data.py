# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import os
import random

import numpy as np
import torch
import tqdm.auto as tqdm
from datasets import load_dataset, load_from_disk


def set_seed(seed):
    """
    Set seed for reproducibility.

    Args:
        seed: int, The seed to set

    Returns:
        None
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)


class TokenizerWrapper:
    """
    Wrapper for tokenized input IDs.
    """

    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load and process WikiText-2 dataset.
    """
    print("Loading WikiText-2 dataset.")
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer):
    """
    Load and process C4 dataset.
    """
    print("Loading C4 dataset.")

    os.makedirs("data", exist_ok=True)

    if os.path.exists("data/c4-train.pt"):
        traindata = load_from_disk("data/c4-train.pt")
        valdata = load_from_disk("data/c4-val.pt")
    else:
        try:
            traindata = load_dataset(
                "allenai/c4",
                "allenai--c4",
                data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                split="train",
            )
            valdata = load_dataset(
                "allenai/c4",
                "allenai--c4",
                data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                split="validation",
            )
        except Exception:
            traindata = load_dataset(
                "allenai/c4",
                data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                split="train",
            )
            valdata = load_dataset(
                "allenai/c4",
                data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                split="validation",
            )

        traindata.save_to_disk("data/c4-train.pt")
        valdata.save_to_disk("data/c4-val.pt")

    random.seed(seed)
    trainloader = []
    progress_bar = tqdm.tqdm(range(nsamples))
    for _ in progress_bar:
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        progress_bar.set_description("Generating Samples")

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_openwebtext(seed, seqlen, tokenizer):
    """
    Load and process OpenWebText dataset.
    """
    print("Loading OpenWebText dataset.")
    raw_datasets = load_dataset("openwebtext")
    raw_datasets = raw_datasets["train"].train_test_split(
        test_size=0.05,
        seed=seed,
        shuffle=True,
    )

    trainloader = None
    valdata = raw_datasets["test"]
    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_slimpajama(nsamples, seed, seqlen, tokenizer):
    """
    Load and process SlimPajama dataset.
    """
    print("Loading SlimPajama dataset.")
    traindata = load_dataset("DKYoon/SlimPajama-6B", split="train")
    testdata = load_dataset("DKYoon/SlimPajama-6B", split="test")

    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    progress_bar = tqdm.tqdm(range(nsamples))
    for _ in progress_bar:
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        progress_bar.set_description("Generating Samples")

    return trainloader, testenc


def get_pile_dm_math(nsamples, seed, seqlen, tokenizer):
    """
    Load and process The Pile DM Mathematics subset.
    Uses ArmelR/the-pile-splitted, which is already separated by subset.
    """
    print("Loading The Pile - DM Mathematics dataset.")

    os.makedirs("data", exist_ok=True)
    cache_train_path = "data/pile-dm-math-train.pt"
    cache_test_path = "data/pile-dm-math-test.pt"

    if os.path.exists(cache_train_path) and os.path.exists(cache_test_path):
        traindata = load_from_disk(cache_train_path)
        testdata = load_from_disk(cache_test_path)
    else:
        traindata = load_dataset(
            "ArmelR/the-pile-splitted",
            "DM Mathematics",
            split="train",
        )
        testdata = load_dataset(
            "ArmelR/the-pile-splitted",
            "DM Mathematics",
            split="test",
        )

        traindata.save_to_disk(cache_train_path)
        testdata.save_to_disk(cache_test_path)

    random.seed(seed)
    trainloader = []
    progress_bar = tqdm.tqdm(range(nsamples))

    for _ in progress_bar:
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] > seqlen:
                break

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        progress_bar.set_description("Generating Samples")

    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    testenc = testenc.input_ids[:, : (256 * seqlen)]
    testenc = TokenizerWrapper(testenc)

    return trainloader, testenc

def get_codeparrot(nsamples, seed, seqlen, tokenizer):
    print("Loading CodeParrot dataset...")
    random.seed(seed)

    data_files = [
        "https://huggingface.co/datasets/codeparrot/github-code/resolve/main/data/train-00000-of-01126.parquet",
        "https://huggingface.co/datasets/codeparrot/github-code/resolve/main/data/train-00001-of-01126.parquet",
        "https://huggingface.co/datasets/codeparrot/github-code/resolve/main/data/train-00002-of-01126.parquet",
        "https://huggingface.co/datasets/codeparrot/github-code/resolve/main/data/train-00003-of-01126.parquet",
        "https://huggingface.co/datasets/codeparrot/github-code/resolve/main/data/train-00004-of-01126.parquet",
    ]

    dataset = load_dataset(
        "parquet",
        data_files={"train": data_files},
        split="train",
        streaming=True,
    )

    trainloader = []
    progress_bar = tqdm.tqdm(total=nsamples)
    data_iter = iter(dataset)

    while len(trainloader) < nsamples:
        try:
            sample = next(data_iter)
        except StopIteration:
            raise RuntimeError(
                f"CodeParrot stream ended early: only collected {len(trainloader)} "
                f"usable samples out of requested {nsamples}."
            )

        text = sample.get("content", None)
        if text is None:
            text = sample.get("code", None)

        if text is None or not isinstance(text, str) or len(text.strip()) == 0:
            continue

        trainenc = tokenizer(text, return_tensors="pt")
        if trainenc.input_ids.shape[1] <= seqlen:
            continue

        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

        progress_bar.update(1)
        progress_bar.set_description("Generating CodeParrot Samples")

    progress_bar.close()

    eval_text = "def hello_world():\n    print('hello world')\n" * 200
    testenc = tokenizer(eval_text, return_tensors="pt")
    testenc = testenc.input_ids[:, : (256 * seqlen)]
    testenc = TokenizerWrapper(testenc)

    return trainloader, testenc


def get_loaders(
    name,
    nsamples=128,
    seed=0,
    seqlen=2048,
    tokenizer=None,
):
    """
    Get loaders for the specified dataset.
    """
    lname = name.lower()

    if "wikitext2" in lname:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "c4" in lname:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    elif "openwebtext" in lname:
        return get_openwebtext(seed, seqlen, tokenizer)
    elif "slimpajama" in lname:
        return get_slimpajama(nsamples, seed, seqlen, tokenizer)
    elif "pile_dm_math" in lname:
        return get_pile_dm_math(nsamples, seed, seqlen, tokenizer)
    elif "codeparrot" in lname:
        return get_codeparrot(nsamples, seed, seqlen, tokenizer)
    else:
        raise ValueError(f"Unknown dataset {name}")


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import lm_eval

    try:
        lm_eval.simple_evaluate(
            model="hf",
            model_args="pretrained=facebook/opt-125m,dtype=half,device=cpu",
            tasks=[
                "arc_easy",
                "arc_challenge",
                "winogrande",
                "openbookqa",
            ],
            verbosity="ERROR",
        )
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    for name in ["wikitext2", "c4", "openwebtext", "slimpajama", "pile_dm_math","codeparrot"]:
        try:
            trainloader, testenc = get_loaders(
                name, nsamples=128, seqlen=1024, tokenizer=tokenizer
            )
            print(f"Loaded {name} successfully.")
        except Exception as e:
            print(f"Failed to load {name}: {e}")