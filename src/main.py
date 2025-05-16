# adapted from https://github.com/Betswish/Cross-Lingual-Consistency/blob/main/1_easyrun/main.py


import json
import re
import os
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch

def predict_mask(answer_cand, prompt, model_type):
    answer_pred_probs = dict()
    
    for answer in answer_cand:
        answer_cand_probs = []
    
        if "qwen" in model_type:
            prompt_new = prompt.replace("<mask>", answer)

            model_input = tokenizer(prompt_new, return_tensors='pt')
            # add token 151643 at the beginning of the input_ids, which is the token for eos_token_id in Qwen. in case the <mask> is at the beginning of the prompt
            model_input['input_ids'] = torch.cat([torch.tensor([[151643]]).to(model_input['input_ids']), model_input['input_ids']], dim=-1)
            model_input = model_input.to(device)

            output = model(**model_input)
            
            logits = output['logits'][0, :-1] 
            token_ids = model_input['input_ids'][0, 1:]
    
            answer_pred_probs[answer] = float(torch.nn.CrossEntropyLoss(reduction='mean')(logits, token_ids))
        else:
            raise ValueError("Model type not supported")
    
    return answer_pred_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--lang',
        default='en',
        help='language',
        type=str,
    )

    parser.add_argument(
        '--model_path',
        help='model path',
        type=str,
    )

    parser.add_argument(
        '--model_type',
        type=str,
    )

    parser.add_argument(
        '--data_path',
        default='data',
    )

    parser.add_argument(
        '--output_prefix',
        default='output',
    )

    args = parser.parse_args()

    lang = args.lang
    model_type = args.model_type
    model_path = args.model_path
    data_path = args.data_path
    output_prefix = args.output_prefix

    record_name = output_prefix + '_Rankings_' + lang + '.json'

    if not os.path.exists(record_name):
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("Runing on:" + device)
        print()
        model = model.to(device)

        data = []
        
        # load json data
        with open(data_path) as f:
            raw_data = json.load(f)
            for item in raw_data:
                # replace "___" with "<mask>" in the question
                prompt = re.sub(r'_+', '<mask>', item['question'])
                if isinstance(item['answer'], list):
                    d = [prompt, item['answer'], item['options']]
                else:
                    d = [prompt, [item['answer']], item['options']]
                data.append(d)

        print("Example:", data[0])

        all_gold_ans = []
        answer_pred_orig_probs = dict()
        
        pred_corr = 0
        pred_tot = 0

        # For saving probing results
        correct_index = []
        answer_count = []
        rank = []
        score_full = []

        for i, d in tqdm(enumerate(data)):
      
            prompt = d[0]
            gold_ans_list = d[1]
            answer_cand = d[2]
            
            all_gold_ans.append(gold_ans_list)

            answer_pred_probs = predict_mask(answer_cand, prompt, model_type)
            sorted_probs = sorted(answer_pred_probs.items(), key=lambda x: x[1], reverse=False)
            
            ranked_keys = [x[0] for x in sorted_probs]
            rank.append(ranked_keys[:])

            score_full.append(answer_pred_probs)

            corr = 0
            tot = len(gold_ans_list)

            correct_gold = [] 
            for gold_ans in gold_ans_list:
                if gold_ans in ranked_keys[:tot]:
                    corr += 1
                    correct_gold.append(1) 
                else: 
                    correct_gold.append(0) 
            
            correct_index.append(correct_gold)
            answer_count.append(tot)


            pred_corr += corr
            pred_tot += tot


        # Saving probing results to files
        with open(output_prefix + '_Rankings_' + lang + '.json', 'w') as f:
            json.dump(rank, f, ensure_ascii=False)

        with open(output_prefix + '_Scores_' + lang + '.json', 'w') as f:
            json.dump(str(score_full), f, ensure_ascii=False, indent=4)

        with open(output_prefix + '_CorrectIndex_' + lang + '.json', 'w') as f:
            json.dump(correct_index, f)
        with open(output_prefix + '_AnswerCount_' + lang + '.json', 'w') as f:
            json.dump(answer_count, f)

        # Print probing accuracy
        print("output_prefix: ", output_prefix)
        print("Accuray in ", lang + ': ', [pred_corr/pred_tot])
        print(pred_corr)
        print(pred_tot)

    else:
        print("Already exists:", record_name)
