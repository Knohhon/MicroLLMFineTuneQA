import evaluate
import collections
import numpy as np
import tqdm



def compute_metrics(start_logits, end_logits, features, examples, n_best=20, max_answer_length=50):
    """Compute the Exact Match (EM) and F1 score for the model's predictions.
    
    Reconstruct the actual text of the answer from the model's predictions and compare
    it to the ground truth for the validation dataset.
    
    Args:
        start_logits: Logits predicting the start position of the answer.
        end_logits: Logits predicting the end position of the answer.
        features: The processed validation dataset.
        examples: The raw validation dataset.
        n_best: The top-k answers to consider.
        max_answer_length: The maximum length of an answer to consider.
    
    Returns:
        The Exact Match (EM) and F1 score for the validation dataset.
    """

    metric = evaluate.load("squad")

    # keep a dictionary that maps examples to predictions through unique IDs
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            # keep a list of the top-k best predictions for the start and end position indexes
            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    # reconstruct the answer considering each prediction for the start and end positions 
                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # select the answer with the best score based on the logit scores
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            # create a list with the predictions that contains the IDs and actual text
            # see: https://huggingface.co/spaces/evaluate-metric/squad
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    # create a list with the labels that contains the IDs and actual text
    # see: https://huggingface.co/spaces/evaluate-metric/squad
    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    metrics = metric.compute(predictions=predicted_answers, references=theoretical_answers)
    print(type(metrics), metrics.keys())

    #with open(csv_path, 'a', newline='', encoding='utf-8') as f:
    #            writer = csv.DictWriter(f, fieldnames=['step', 'epoch', *logs.keys()])
    #            writer.writerow({'step': state.global_step, 'epoch': state.epoch, **logs})
    
    return metrics