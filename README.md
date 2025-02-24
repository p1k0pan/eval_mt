# eval_mt

指标：BLEU, CHRF, CHRF++, TER, BERT-SCORE, METEOR, COMET

## 安装

```
pip install jieba bert-score sacrebleu torch pandas unbabel-comet nltk
```

其他语象
```
pip install "sacrebleu[ja]" "sacrebleu[ko]"
pip install unidic-lite
```

## OCR指标
1. 接受json格式
```json
{
    "en/25/32/en_10101166.jpg": {
        "ref": "ALUSSO\nCRI 80+\nOthers\nCRI 75+",
        "output": "ALUSSO\nCRI 80+\n\nOthers\nCRI 75-"
    },
}
```
ref 为标准答案，output为模型输出。

也可接受output或ref为列表形式的json
```json
{
    "OCRMT30K_06269.jpg": {
        "output": "华丰 三鲜伊面\n我们做好面 您可以信赖\n和面用高汤\n蔬菜更加量\n50%",
        "ref": [
            "和面用高汤",
            "@北京人不知道的北京事儿",
            "华丰",
            "三鲜伊面",
            "蔬菜更加量"
        ]
    },
}
```

2. 在代码中替换需要测试的文件夹以及语言

在`folders`提供翻译的json文件的父级目录后，`pathlib`可以将文件夹下的所有json文件都遍历，不需要每一个json文件单独输入测试。

需要在`lang`中指定目标语言是什么。目前支持zh, en, ja, ko

3. 输出的文件

如果翻译的文件是`original.json`，那么测评完会生成
- `orginal_ocr_eval.json`：计算的是每一个句子的TP, FP, FN，以及每一个图片对应的Precision, Recall, F1。
- `ocr_overall.csv`：计算的是一个目录下，每个json文件各自的总体Precision, Recall, F1。
- TP：模型识别 且 在标准答案里
- FP：模型识别 但 不在标准答案里
- FN：模型没识别 但 标准答案有
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * precision * recall / (precision + recall)

## 翻译指标
1. 接受json格式的翻译结果如下：
```json
{
    "en_194.jpg": {
        "src": "Vacant",
        "ref": "空闲",
        "mt": "空闲"
    },
    "en_26.jpg": {
        "src": "TSUNAMA\nEVACUATION\nROUTE",
        "ref": "海啸\n撤离\n路线",
        "mt": "海啸\n疏散路线\n路线"
    },
}
```
src为原文，ref为标准答案，mt为模型翻译。拼接用`\n`或空格都行，只要三个文本都用同一种方式就行。

2. 在代码中替换需要测试的文件夹以及语言

在`folders`提供翻译的json文件父级目录后，`pathlib`可以将文件夹下的所有json文件都遍历，不需要每一个json文件单独输入测试。

需要在`lang`中指定目标语言是什么。支持zh, ja, ko, en

3. 输出的文件

如果翻译的文件是`original.json`，那么测评完会生成
- `orginal_total.csv`：计算的是corpus_bleu
- `original_each.csv`：每一个图片单独的bleu值
- `original_each_avg.csv`：计算的是sentence_bleu（每个图片单独算bleu值最后取平均）
