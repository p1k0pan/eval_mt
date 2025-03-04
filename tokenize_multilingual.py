import nltk
import pythainlp
from indicnlp.tokenize import sentence_tokenize, indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

import MeCab
import mecab_ko as MeCab_ko
import jieba
from hazm import Normalizer, WordTokenizer
import spacy
from zemberek import TurkishTokenizer
import logging

# 禁用 pymorphy3 的日志
logging.getLogger("pymorphy3").setLevel(logging.ERROR)

ru_nlp = spacy.load("ru_core_news_sm")
wakati = MeCab.Tagger("-Owakati")
tagger_ko = MeCab_ko.Tagger("-Owakati")
ar_normalizer = Normalizer()
ar_tokenizer = WordTokenizer()
tr_tokenizer = TurkishTokenizer.DEFAULT

def tokenize_ru(text):
    """
    """

    doc = ru_nlp(text)
    tokens = [token.text.strip() for token in doc]
    return tokens

def tokenize_ar(text):
    """
    阿拉伯语分词示例：这里用 hazm 或 Camel Tools（camel_tools）。
    以 hazm 为例。
    """
    
    text_norm = ar_normalizer.normalize(text)
    tokens = ar_tokenizer.tokenize(text_norm)
    return tokens

def tokenize_th(text):
    """
    泰语分词示例：使用 pythainlp。
    """
    tokens = pythainlp.word_tokenize(text)  # 默认新mm分词器
    return tokens

def tokenize_hi(text):
    """
    印地语分词示例：使用 indic_nlp_library。
    这里仅演示调接口，安装和初始化可能需要额外步骤（如下载资源等）。
    """
    tokens = indic_tokenize.trivial_tokenize(text.strip())
    return tokens

def tokenize_tr(text):

    
    tokens = tr_tokenizer.tokenize(text)
    res_tokens = [t.content for t in tokens]
    return res_tokens

def tokenize_zh(text):
    """
    中文分词示例：使用 jieba。
    """
    tokens = jieba.cut(text, cut_all=False)
    return list(tokens)

def tokenize_ja(text):
    tokens_str = wakati.parse(text)
    tokens = tokens_str.strip().split()
    return tokens

def tokenize_ko(text):
    tokens_str = tagger_ko.parse(text)
    tokens = tokens_str.strip().split()
    return tokens

def tokenize_default(text):
    """
    对于未特殊处理的语言（比如英语、法语、德语、西班牙语等），
    可以先尝试 nltk.word_tokenize 或空格切分。
    """
    return nltk.word_tokenize(text)

if __name__ == "__main__":
    # text = "Легкий\nМягкий\nСтиральный\nЭргономичный\nПротивоскользящий\nБез шума"
    # print(tokenize_ru(text))
    text = "लाइटवेट, सॉफ्ट, वशेश्य, एरगोनॉमिक्स, अंटी-स्लिप, नो नोइज"
    print(tokenize_hi(text))