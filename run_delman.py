import torch
import shutil
import argparse
from itertools import islice
from time import time
from typing import Tuple, Union

from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import MultiHarmbenchDataset


from delman import MEMITHyperParams, apply_delman_to_model
from util import nethook
from util.globals import *

ALG_DICT = {
    "DELMAN": (MEMITHyperParams, apply_delman_to_model),
}

DS_DICT = {
    "HarmBench": (MultiHarmbenchDataset),
}

def main(
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    conserve_memory: bool,
    data_name:str,
    model_path:str = None,
    num_batch: int = 0,
    save_model: bool = True,
    out_name: str = None
):
    
    # Determine run directory
    run_dir = RESULTS_DIR / out_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = HPARAMS_DIR / hparams_fname
    hparams = MEMITHyperParams.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing DELMAN with parameters {hparams}")

    # Instantiate vanilla model
    if model_path is not None:
        model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16).cuda()
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.pad_token = tok.eos_token
        tok.add_bos_token = False
        tok.padding_side = 'right'

    elif type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
        tok.add_bos_token = False
        tok.padding_side = 'right'
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load dataset
    ds_class = DS_DICT[ds_name]

    ds = ds_class(DATA_DIR, tok=tok, size=dataset_size_limit, data_name = data_name)
    if num_batch == 0:
        num_edits = len(ds) 
    else:
        num_edits = len(ds) // num_batch + (1 if len(ds) % num_batch != 0 else 0)
        
    print(f'Edits model with {num_batch} incremental batches, {num_edits} datas in each batch')

    edited_model = model


    start = time()
    for i,record_chunks in enumerate(chunks(ds, num_edits)):
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        
        edited_model, weights_copy = apply_delman_to_model(
            edited_model,
            tok,
            [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
        )
        print("***************************************")
        print("i: ", i,"/", len(ds))
        print("***************************************")

    all_exec_time = time() - start
    print("all time is", all_exec_time)
    if save_model:
        print('save the model to ', run_dir)
        edited_model.save_pretrained(run_dir)
        tok.save_pretrained(run_dir)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to apply DELMAN.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        default="Qwen2.5-7B-Instruct.json",
        type=str,
        help="Name of hyperparameters file, located in the hparams folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        default="HarmBench",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--data_name",
        default="HarmBench.json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path of model and tokenizer"
    )
    parser.add_argument(
        "--num_batch",
        type=int,
        default=200,
        help="Number of batches.",
    )
    parser.add_argument(
        "--save_model",
        action='store_true',
        default=True,
        help='whether to save the model after edition'
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default=None,
        help='the out dir name'
    )
    parser.set_defaults(conserve_memory=False)
    args = parser.parse_args()
    main(
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.conserve_memory,
        data_name = args.data_name,
        model_path=args.model_path,
        num_batch = args.num_batch,
        save_model = args.save_model,
        out_name = args.out_name
    )
