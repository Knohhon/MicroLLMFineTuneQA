def preprocess_train_examples(examples, tokenizer, max_length, stride):
    """Process the training split of the SQuAD dataset.
    
    Process the training split of the SQuAD dataset to include tokenized questions
    and context, as well as the start and end positions of the answer within the context.
    
    Args:
        examples: A row from the dataset containing an example.
        tokenizer: The BERT tokenizer to be used.
        max_length: The maximum length of the input sequence. If exceeded, truncate the second
            sentence of a pair (or a batch of pairs) to fit within the limit.
        stride: The number of tokens to retain from the end of a truncated sequence, allowing
            for overlap between truncated and overflowing sequences.
    
    Returns:
        The processed example.
    """
    # Tokenize the questions and context sequences
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
      questions,
      examples["context"],
      truncation="only_second",
      padding="max_length",
      stride=stride,
      max_length=max_length,
      return_offsets_mapping=True,
      return_overflowing_tokens=True,
    )

    answers = examples["answers"]
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")

    start_positions = []
    end_positions = []

    # find the start and end positions of the answer within the context
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # if the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


def preprocess_valid_examples(examples, tokenizer, max_length, stride):
    """Process the validation split of the SQuAD dataset.
    
    Process the training split of the SQuAD dataset to include the unique ID of each row,
    the tokenized questions and context, as well as the start and end positions of the answer
    within the context.
    
    Args:
        examples: A row from the dataset containing an example.
        tokenizer: The BERT tokenizer to be used.
        max_length: The maximum length of the input sequence. If exceeded, truncate the second
            sentence of a pair (or a batch of pairs) to fit within the limit.
        stride: The number of tokens to retain from the end of a truncated sequence, allowing
            for overlap between truncated and overflowing sequences.
    
    Returns:
        The processed example.
    """
    # Tokenize the questions and context sequences
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
      questions,
      examples["context"],
      truncation="only_second",
      padding="max_length",
      stride=stride,
      max_length=max_length,
      return_offsets_mapping=True,
      return_overflowing_tokens=True,
    )
    
    example_ids = []
    answers = examples["answers"]
    offset_mapping = inputs["offset_mapping"]
    sample_map = inputs.pop("overflow_to_sample_mapping")

    start_positions = []
    end_positions = []

    # find the start and end positions of the answer within the context
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # if the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["example_id"] = example_ids  # keep the unique ID of the example
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs