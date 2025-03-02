from ast import Not
import json
from pathlib import Path
import pandas as pd
import os
import jieba
import sys
import mecab_ko as MeCab_ko
import MeCab

def eval_ocr(mt_file, lang):
    mt = json.load(open(mt_file, "r"))
    # 用于存储每个句子的指标结果
    results = {}

    # 综合统计指标
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_ref_words = 0
    tagger_ko = MeCab_ko.Tagger("-Owakati")
    wakati = MeCab.Tagger("-Owakati")

    # 遍历所有图片的 OCR 结果
    for img, item in mt.items():
        ocr_ref1 = item["ref"]
        ocr_mt1 = item["output"]
        if lang == "zh":
            if type(ocr_ref1) == list:
                ocr_ref1 = [word for word in ocr_ref1 if word != "###"]
                ocr_ref = jieba.cut(" ".join(ocr_ref1), cut_all=False)  # 参考文本（单词分割）列表
            else:
                ocr_ref = jieba.cut(ocr_ref1, cut_all=False)  # 参考文本（单词分割）列表
            if type(ocr_mt1) == list:
                ocr_mt = jieba.cut(" ".join(ocr_mt1), cut_all=False)
            else:
                ocr_mt = jieba.cut(ocr_mt1, cut_all=False)  # 模型输出（单词分割）
            ocr_ref=list(ocr_ref)
            ocr_mt=list(ocr_mt)

        elif lang == "ko":
            if type(ocr_ref1) == list:
                ocr_ref = tagger_ko.parse(" ".join(ocr_ref1)).split()  # 参考文本（单词分割）列表
            else:
                ocr_ref = tagger_ko.parse(ocr_ref1).split()  # 参考文本（单词分割）列表
            if type(ocr_mt1) == list:
                ocr_mt = tagger_ko.parse(" ".join(ocr_mt1)).split()
            else:
                ocr_mt = tagger_ko.parse(ocr_mt1).split()

        elif lang == "ja":
            if type(ocr_ref1) == list:
                ocr_ref = wakati.parse(" ".join(ocr_ref1)).split()  # 参考文本（单词分割）列表
            else:
                ocr_ref = wakati.parse(ocr_ref1).split()  # 参考文本（单词分割）列表
            if type(ocr_mt1) == list:
                ocr_mt = wakati.parse(" ".join(ocr_mt1)).split()
            else:
                ocr_mt = wakati.parse(ocr_mt1).split()
            
        else: # en
            if type(ocr_ref1) == list:
                ocr_ref = " ".join(ocr_ref1).split()  # 参考文本（单词分割）列表
            else:
                ocr_ref = ocr_ref1.split()  # 参考文本（单词分割）字符串
            if type(ocr_mt1) == list:
                ocr_mt = " ".join(ocr_mt1).split()
            else:
                ocr_mt = ocr_mt1.split()  # 模型输出（单词分割）

        ocr_ref = [x for x in ocr_ref if x not in (' ', '\n')]
        ocr_mt = [x for x in ocr_mt if x not in (' ', '\n')]
        # 计算 TP, FP, FN
        tp = [word for word in ocr_mt if word in ocr_ref]  # 模型输出正确的单词
        fp = [word for word in ocr_mt if word not in ocr_ref]  # 模型多余的单词
        fn = [word for word in ocr_ref if word not in ocr_mt]  # 模型遗漏的单词

        # 计算 Precision, Recall, F1-Score 和 Accuracy
        precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
        recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
        f1_score = (
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        )
        accuracy = len(tp) / len(ocr_ref) if len(ocr_ref) > 0 else 0

        # 存储结果
        results[img] = {
            "ref": item["ref"],
            "mt": item["output"],
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score,
            "Accuracy": accuracy,
        }

        # 更新总计指标
        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)
        total_ref_words += len(ocr_ref)

    # 计算综合指标
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1_score = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )
    overall_accuracy = total_tp / total_ref_words if total_ref_words > 0 else 0

    json.dump(results, open(mt_file.with_name(mt_file.stem + "_ocr_eval.json"), "w"), indent=4, ensure_ascii=False)

    return {
        "Precision": overall_precision,
        "Recall": overall_recall,
        "F1-Score": overall_f1_score,
        "Accuracy": overall_accuracy,
    }



if __name__ == "__main__":
    folders = [
        # "ocr_tool/mit10", 
        # "ocr_tool/anytrans/en"
        # "ocr_tool/ocrmt", 
        # "ocr_tool/anytrans/zh"
        # "ocr_tool/anytrans/ko"
        # "ocr_tool/anytrans/ja"
        # "ocr_tool/lsvt_full"
        "ocr_tool/icdar2015"
        ]
    lang="en" # zh, en, ja, ko

    for folder in folders:
        folder = Path(folder)
        overall=[]
        for file in folder.rglob("*.json"):
            if os.path.exists(folder / f"{file.stem}_ocr_eval.json") or file.stem.endswith("_ocr_eval"):
                continue
            print("processing:", file)
            metrics = eval_ocr(file, lang)
           
            metrics["model"] = file
            overall.append(metrics)
            df = pd.DataFrame(overall)
            df.to_csv(folder / "ocr_overall.csv", index=False, encoding='utf-8-sig')