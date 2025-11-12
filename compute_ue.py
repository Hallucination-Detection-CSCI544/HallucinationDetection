import os
import pandas as pd, json, numpy as np
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

if not hasattr(transformers, "AdamW"):
    transformers.AdamW = torch.optim.AdamW

from lm_polygraph.utils.model import WhiteboxModel
from lm_polygraph.estimators import MeanTokenEntropy, MaximumSequenceProbability, MaximumTokenProbability, TokenEntropy
from lm_polygraph.utils.manager import estimate_uncertainty  # or UEManager
from lm_polygraph.utils.generation_parameters import GenerationParameters

import matplotlib.pyplot as plt
import seaborn as sns




def read_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    cols = [
        
        "clean_prompt", "golden_answer", "wrong_answer",
        "golden_answer_token", "wrong_answer_token",
        "prompt_with_bad_shots/alice", "count_knowledge", "-1"
    ]
    df = pd.DataFrame(data, columns=cols).head(50)
    print(df.head(1))
    return df

if __name__ == "__main__":
    general_path = "datasets/AliceGeneralTrivia_qa_no_contextWithThreshold1.0_mistralai_Mistral-7B-v0.3.json"
    hallucinate_path = "datasets/AliceHallucinateTrivia_qa_no_contextWithThreshold1.0_mistralai_Mistral-7B-v0.3.json"
    non_hallucinate_path = "datasets/AliceNonHallucinateTrivia_qa_no_contextWithThreshold1.0_mistralai_Mistral-7B-v0.3.json"

    general_df = read_dataset(general_path)
    hallucinate_df = read_dataset(hallucinate_path)
    non_hallucinate_df = read_dataset(non_hallucinate_path)

    model_path = "mistralai/Mistral-7B-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0")
    # propagate pad id to model + generation config to avoid warnings
    base_model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(base_model, "generation_config") and base_model.generation_config is not None:
        base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    gen_params = GenerationParameters(
        do_sample=False,
        max_new_tokens=5,
        )
    model = WhiteboxModel(base_model, tokenizer, generation_parameters=gen_params)
    ue_method = MaximumTokenProbability()
    def run(df):
        ues = []
        for p in df["prompt_with_bad_shots/alice"]:
            ue = estimate_uncertainty(model, ue_method, p)
            print(ue)
            if len(ue.generation_tokens) > 1 and (ue.generation_tokens[0] == 1183 or ue.generation_tokens[0] == 1098):
                unc = ue.uncertainty[1] * -1
            else:
                unc = ue.uncertainty[0] * -1
            ues.append(unc)
        return ues

    general_df["ue"] = run(general_df)
    hallucinate_df["ue"] = run(hallucinate_df)
    non_hallucinate_df["ue"] = run(non_hallucinate_df)

    print(f"Average UE for HK-: {np.mean(general_df['ue'])}")

    print(f"Average UE for HK+: {np.mean(hallucinate_df['ue'])}")

    print(f"Average UE for Correct: {np.mean(non_hallucinate_df['ue'])}")


    general_df.to_csv("general_df.csv", index=False)
    hallucinate_df.to_csv("hallucinate_df.csv", index=False)
    non_hallucinate_df.to_csv("non_hallucinate_df.csv", index=False)

    
    plot_df = pd.DataFrame({
        "condition": (
            ["HK-"] * len(general_df["ue"])
            + ["HK+"] * len(hallucinate_df["ue"])
            + ["Correct"] * len(non_hallucinate_df["ue"])
        ),
        "uncertainty": (
            general_df["ue"].tolist()
            + hallucinate_df["ue"].tolist()
            + non_hallucinate_df["ue"].tolist()
        )
    })


    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="condition", y="uncertainty", data=plot_df,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black"},
    )

    plt.ylim(0.0, 1.0)
    plt.title("Certainty Distribution by Condition (Alice-Bob)")
    plt.xlabel("Condition")
    plt.ylabel("Certainty")
    plt.tight_layout()
    plt.savefig("uncertainty_boxplot.png", dpi=300)
    plt.close()

    print("Normalized box plot saved as 'uncertainty_boxplot.png'")