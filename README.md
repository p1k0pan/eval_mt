# eval_mt

指标：BLEU, CHRF, CHRF++, TER, BERT-SCORE, METEOR, COMET

## 安装

```
pip install jieba bert-score sacrebleu torch pandas unbabel-comet nltk
```

其他语象
```
pip install "sacrebleu[ja]" "sacrebleu[ko]"
```

## 使用
1. 接受json格式的翻译结果如下：
```json
[
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
]
```
src为原文，ref为标准答案，mt为模型翻译。拼接用`\n`或空格都行，只要三个文本都用同一种方式就行。

2. 在代码中替换需要测试的文件夹以及语言

在`folders`提供翻译的json文件后，`pathlib`可以将文件夹下的所有json文件都遍历，不需要每一个json文件单独输入测试。

需要在`lang`中指定目标语言是什么。支持zh, ja, ko, en

3. 输出的文件

如果翻译的文件是`original.json`，那么测评完会生成
- `orginal_total.csv`：计算的是corpus_bleu
- `original_each.csv`：每一个图片单独的bleu值
- `original_each_avg.csv`：计算的是sentence_bleu（每个图片单独算bleu值最后取平均）
