# Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon

Official code repository for the ACL'25 paper "Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon".

## Data
We provide the cultural probing questions used in our experiments, consisting of English/non-English cultural questions about the four non-Anglophonic communities (Koreans `ko` in South Korea, Han Chinese `zh`, Tibetans `bo`, and Mongols `mn` in China).

The data is are available in the `data` folder.

## Code
We provide the code for evaluating the probing questions in the `src` folder. 
It is adapted from the code of the paper ["Cross-Lingual Consistency of Factual Knowledge in Multilingual Language Models"](https://github.com/Betswish/Cross-Lingual-Consistency)

Here is an example command to run the code:
```bash
cd src

python main.py \
--lang en \
--data_path ../data/bo-culture-cloze_en.json \
--model_type qwen \
--model_path {model_path} \
--output_prefix ../output/{model_name}_bo-culture-cloze_en
```


# Citation
TBD
