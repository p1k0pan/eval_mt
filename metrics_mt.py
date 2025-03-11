import json
from pathlib import Path
import pandas as pd
import os
import jieba
import sys

import sacrebleu
from transformers import AutoTokenizer
# from sacrebleu.metrics import BLEU, CHRF, TER
from bert_score import score
import json
import sys
from nltk.translate import meteor_score
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
from comet import download_model, load_from_checkpoint
from tokenize_multilingual import *
model_path = download_model("Unbabel/wmt22-comet-da")

# Load the model checkpoint:
comet_model = load_from_checkpoint(model_path)

def bleu_score(predict, answer, lang, is_sent=False):
    """
    refs = [ 
             ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'], 
           ]
    sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']
    """
    tokenize_map = {
        'zh': "zh",
        'ja': "ja-mecab",
        'ko': "ko-mecab",
        'th': "none",    # 泰语使用 Flores101 分词
        'ar': "none",     # 阿拉伯语
        'hi': "none",     # 印地语
        'ru': "none",            # 俄语专用规则
        'tr': "none",            # 土耳其语专用规则
        'de': "intl",            # 德语专用规则
        'fr': "intl",            # 法语专用规则
        'es': "intl",            # 西班牙语专用规则
        'it': "intl",            # 意大利语专用规则
        'pt': "intl",            # 葡萄牙语专用规则
    }
    # bleu = sacrebleu.corpus_bleu(predict, answer, lowercase=True, tokenize="flores101")
    tokenize = tokenize_map.get(lang, "13a")
    tokenizer_func = None
    if lang == "ar":
        tokenizer_func = tokenize_ar
    elif lang == "ru":
        tokenizer_func = tokenize_ru
    elif lang == "th":
        tokenizer_func = tokenize_th
    elif lang == "hi":
        tokenizer_func = tokenize_hi
    elif lang == "tr":
        tokenizer_func = tokenize_tr
    if tokenizer_func is not None:
        predict = [" ".join(tokenizer_func(p)) for p in predict]
        answer = [[" ".join(tokenizer_func(a)) for a in answer[0]]]

    if is_sent:
        bleu = sacrebleu.sentence_bleu(predict, answer, lowercase=True, tokenize=tokenize)
    else:
        bleu = sacrebleu.corpus_bleu(predict, answer, lowercase=True, tokenize=tokenize)
    return bleu.score

def chrf_score(predict, answer):
    chrf = sacrebleu.corpus_chrf(predict, answer)
    return chrf.score

def chrfppp_score(predict, answer):
    
    chrfppp = sacrebleu.corpus_chrf(predict, answer, word_order=2)
    return chrfppp.score

def ter_score(predict, answer):
    ter = sacrebleu.corpus_ter(predict, answer, asian_support=True, normalized=True, no_punct=True)
    return ter.score

def bertscore(predict, answer, lang):
    P, R, F1 = score(predict, answer, lang=lang, device="cuda")
    return torch.mean(P).item(), torch.mean(R).item(), torch.mean(F1).item()

def meteor(predict, answer, type, lang):
    all_meteor = []
    if lang == "zh":
        tokenizer_func = tokenize_zh
    elif lang == "ar":
        tokenizer_func = tokenize_ar
    elif lang == "ru":
        tokenizer_func = tokenize_ru
    elif lang == "th":
        tokenizer_func = tokenize_th
    elif lang == "hi":
        tokenizer_func = tokenize_hi
    elif lang == "tr":
        tokenizer_func = tokenize_tr
    elif lang == "ja":
        tokenizer_func = tokenize_ja
    elif lang == "ko":
        tokenizer_func = tokenize_ko
    else:
        tokenizer_func = tokenize_default
    for i in range(len(predict)):
        ref_tokens = tokenizer_func(answer[i])
        hyp_tokens = tokenizer_func(predict[i])

        score_val = meteor_score.meteor_score([ref_tokens], hyp_tokens)
        all_meteor.append(score_val)
    if type == "total":
        return sum(all_meteor) / len(all_meteor)
    else:
        return all_meteor[0]

def cal_total_metrics(predicts, answers, chrf_10, comet_sys_score, lang):
    bs = bleu_score(predicts, [answers], lang, is_sent=False)
    cs = chrf_score(predicts, [answers])
    cspp = chrfppp_score(predicts, [answers])
    ts = ter_score(predicts, [answers])
    p, r, f1 = bertscore(predicts, answers, lang)
    m = meteor(predicts, answers, "total", lang)
    print("BLEU:", bs)
    print("CHRF:", cs)
    print("TER:", ts)
    print("BERT-P:", p, "BERT-R:", r, "BERT-F1:", f1)
    print("METEOR:", m)
    print("COMET:", comet_sys_score)

    res = [{"BLEU": bs, "CHRF": cs, "CHRF++": cspp, "TER": ts, "BERT-P": p, "BERT-R": r, "BERT-F1": f1, "METEOR": m, "CHRF<10": chrf_10, "COMET": comet_sys_score}]
    df = pd.DataFrame(res)
    df.to_csv(file.with_name(file.stem + "_total.csv"), index=False, encoding='utf-8-sig' )


def cal_each_metrics(predicts, answers, source, comets, lang, img):
    model_output = comet_model.predict(comets, batch_size=8, gpus=1)
    score = model_output.scores
    sys_score= model_output.system_score

    all_result = []
    chrf_10 = 0
    for i in tqdm(range(len(predicts))):
        ans= answers[i]
        pred = predicts[i]
        bs = bleu_score([pred], [[ans]], lang, is_sent=True) 
        cs = chrf_score([pred], [[ans]])
        cspp = chrfppp_score([pred], [[ans]])
        if cs<10:
            chrf_10+=1
        ts = ter_score([pred], [[ans]])
        p, r, f1 = bertscore([pred], [ans], lang)
        m = meteor([pred], [ans], "each", lang)
        all_result.append({"img":img, "reference": ans, "predicts": pred, "source":source[i], "BLEU": bs, "CHRF": cs, "CHRF++": cspp, "TER": ts, "BERT-P": p, "BERT-R": r, "BERT-F1": f1, "METEOR": m, "COMET": score[i]})
    df = pd.DataFrame(all_result)
    df.to_csv(file.with_name(file.stem + "_each.csv"), index=False, encoding='utf-8-sig')
    print("CHRF<10:", chrf_10)
    average_scores = df[["BLEU", "CHRF", "CHRF++", "TER", "BERT-P", "BERT-R", "BERT-F1", "METEOR", "COMET"]].mean()
    average_scores["CHRF<10"] = chrf_10
    avg_df = pd.DataFrame([average_scores])

    avg_df.to_csv(file.with_name(file.stem + "_each_avg.csv"), index=False, encoding='utf-8-sig')
    return chrf_10, sys_score


def eval_line(mt_file, lang):
    mt = json.load(open(mt_file, "r"))
    # 用于存储每个句子的指标结果
    results = {}

    # 遍历所有图片的 OCR 结果
    refs=[]
    mts = []
    comets=[]
    srcs = []
    for img, item in mt.items():
        refs.append(item["ref"])
        mts.append(item["mt"])
        srcs.append(item["src"])
        comets.append({"src": item["src"], "mt": item["mt"], "ref": item["ref"]})
    print("cal each metrics")
    chrf_10, comet_sys_score = cal_each_metrics(mts, refs,srcs, comets, lang, img)
    print("cal total metrics")
    cal_total_metrics(mts, refs, chrf_10, comet_sys_score, lang)


if __name__ == "__main__":
    folders = ["folder1", "folder2"]
    lang = "zh" # zh, en, ko, ja


    for folder in folders:
        folder= Path(folder)
        overall=[]
        for file in folder.rglob("*.json"):
            # if os.path.exists(folder / f"{file.stem}_total.csv"):
            #     continue
            print("processing:", file)
            eval_line(file, lang)