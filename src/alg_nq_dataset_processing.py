import json
import os
import gzip
from typing import List, Dict
from dotenv import load_dotenv
import orjson


def open_gzip_jsonl_data(path):
    data = []
    with gzip.open(path, 'r') as f:
        for line in f:
            _ = orjson.loads(line) if os.getenv('USE_ORJSON') else json.loads(line)
            data.append(_)
    return data


def document_cleaning(document_tokens: List[Dict]):
    return [
        token_dict['token'] 
        for token_dict in document_tokens 
        if not token_dict.get('html_token', False)
    ]

def answer_processing(document: Dict):
    answers = []
    for annotation in document['annotations']:
        if annotation['long_answer']['start_token'] == -1:
            continue
        long_answer = annotation['long_answer']
        long_answer['content'] = [
            token_dict['token']
            for token_dict in document['document_tokens'][long_answer['start_token']:long_answer['end_token']]
            if not token_dict.get('html_token', False)
        ]
        if annotation['short_answers'] == []:
            continue
        short_answers = [ _ for _ in annotation['short_answers'] ]
        for short_answer in short_answers:
            short_answer['content'] = [
                token_dict['token']
                for token_dict in document['document_tokens'][short_answer['start_token']:short_answer['end_token']]
                if not token_dict.get('html_token', False)
            ]
        answers.append({'long_answer': long_answer, 'short_answers': short_answers})
    return answers


def documents_processing(nq_batch : List[Dict]):
    preprocessed_documents = []
    for doc in nq_batch:
        preprocessed_doc = {}
        if doc['document_tokens'] and doc['question_tokens'] and doc['annotations']:
            answers = answer_processing(doc)
            if answers == []:
                continue
            context = document_cleaning(doc['document_tokens'])
            preprocessed_doc['question'] = doc['question_tokens']
            preprocessed_doc['context_base_tokens'] = context
            preprocessed_doc['context_text_in_row'] = ' '.join(context)
            preprocessed_doc['answers'] = answers
            preprocessed_documents.append(preprocessed_doc)

    return preprocessed_documents


def nq_processing(dir : str = os.getenv('RAW_DATASET_DIR_PATH')):
    list_files = os.listdir(dir)
    print(list_files)
    all_documents = []
    for file_name in list_files:
        file_path = os.path.join(dir, file_name)
        nq_batch = open_gzip_jsonl_data(file_path)
        documents = documents_processing(nq_batch)
        all_documents += documents
    return all_documents


def save_jsonl(
    filepath: str = None,
    encoding: str = 'utf-8',
    data: List[Dict] = None
    ):
    with open(filepath, 'w', encoding=encoding) as f:
        line_data = json.dumps(data)
        f.write(line_data + '\n')
    print(f'Dataset save complete, path: {filepath}')


def main():
    all_documents = nq_processing(dir = os.getenv('RAW_DATASET_DIR_PATH'))
    print(len(all_documents))
    save_jsonl(data=all_documents, filepath=os.path.join(os.getenv('PREPROCESSED_DATASET_DIR_PATH'), 'train_nq.jsonl'))



if __name__=='__main__':
    load_dotenv()
    main()