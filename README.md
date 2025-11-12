# Token-Efficient Question Answering with Adaptive RAG via Hallucination Detection
## Dataset Description
The "datasets" folder contains the prompts/answers from TriviaQA and NaturalQA identified in Simhi et al. (2025), "Distinguishing Ignorance from Error in LLM Hallucinations" for the HK-, HK+, and Correct conditions across 3 different models (gemma-9b, llama-8b, and mistral-7b) using the WACK protocol.
* HK-: The HK- samples for a given model are the questions that the LLM does not know the answer to and will thus always answer incorrectly without additional context. These hallucinations can only be mitigated with context retrieval.
* HK+: The HK+ samples for a given model are the questions that the LLM got correct initially, but hallucinated in a setting where noise/errors were injected into the prompt. The two main settings are "snowballing", where erroneous question/answer pairs are prepended to the prompt, and "Alice-Bob" which uses persuasion and text perturbations to induce error. These hallucinations can be mitigated without context retrieval by "fixing" the prompt.
* Correct: The Correct samples are the questions that the LLM got correct in all settings.

In the "datasets" folder, each .json corresponds to the samples for a unique combination of the following conditions:  
* Prompt error setting: jsons beginning with "Alice" were exposed to the Alice-Bob prompts, the others were exposed to the snowballing prompts.
* WACK condition: jsons with "General" in the name are HK- samples. jsons with "Hallucinate" in the name are HK+ samples. jsons with "NonHallucinate" in the name are Correct samples.
* Dataset: jsons with "Trivia_qa" are samples from the TriviaQA dataset and jsons with "Natural_qa" are samples from the NaturalQA dataset.
* Model: jsons will have gemma, llama, or mistral in their name depending on which model these samples were identified for.

So, the samples from TriviaQA that mistral got correct initially but hallucinated on in the snowballing condition would be in the json named "HallucinateTrivia_qa_no_contextWithThreshold1.0_mistralai_Mistral-7B-v0.3.json".

Each json has the following keys/columns: 
* "clean_prompt": Original prompt/question
* "golden_answer": Correct answer from the QA dataset
* "wrong_answer": I think this is a/the wrong answer that the model produced when it hallucinated? This column is still filled out in the non-hallucinate jsons though so I'm not sure how to interpret this column in those files, but regardless I don't think this column is relevant to us
* "golden_answer_token": Token indices corresponding to the ground truth answer for the given model's tokenizer
* "wrong_answer_token": Token indices corresponding to the wrong answer for the given model's tokenizer
* "prompt_with_bad_shots/alice": Altered/noisy prompt, either a snowballing prompt or alice-bob prompt depending on the json
* "count_knowledge": Not totally sure, I think this is the number of times the model got this question correct? A sample is labeled HK+ with even one error so this could be used as a scale for whether an HK+ sample is "closer" to the HK- vs. Correct condition
* "-1": Don't know why this column is here

See read_dataset(path) in compute_ue.py for an example of loading one of the json files into a pandas DataFrame.

Original Paper and Code for Paper:
* https://arxiv.org/abs/2410.22071
* https://github.com/technion-cs-nlp/hallucination-mitigation/tree/main/Distinguishing_Ignorance_from_Error_in_LLM_Hallucinations
