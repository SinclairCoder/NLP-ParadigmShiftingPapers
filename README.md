# NLP-ParadigmShiftingPapers
Reading List of Paradigm Shifting related papers in NLP

![](https://img.shields.io/badge/PRs-welcome-brightgreen) ![](https://img.shields.io/github/stars/sinclaircoder/NLP-ParadigmShiftingPapers?style=social)

## Table of Contents
- [QA-Paradigm](#1-qa-paradigm)
- [Generation-Paradigm](#2-generation-paradigm)
- [Span-Based-Paradigm](#3-span-based-paradigm)
- [Dependency Parsing-Paradigm](#4-dependency-parsing-paradigm)
- [Semantic Segmentation-Paradigm](#5-semantic-segmentation-paradigm)
- [Text Matching-Paradigm](#6-text-matching-paradigm)

## 1. QA-Paradigm

- **The Natural Language Decathlon: Multitask Learning as Question Answering. 2018**  By Salesforce  [[arxiv]](https://arxiv.org/abs/1806.08730) [[code]](https://github.com/salesforce/decaNLP)​

### Sentiment Analysis

- **Sentiment classification towards question-answering with hierarchical matching network.** EMNLP 2018. [[paper]](https://aclanthology.org/D18-1401/) [[code]](https://github.com/clshenNLP/QASC)
  Chenlin Shen, Changlong Sun, Jingjing Wang, Yangyang Kang, Shoushan Li, Xiaozhong Liu, Luo Si, Min Zhang, and Guodong Zhou.


### Aspect-Based Sentiment Analysis 

- **Question-Driven Span Labeling Model for Aspect–Opinion Pair Extraction**.  AAAI 2021 [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17523)
  Lei Gao, Yulong Wang, Tongcun Liu, Jingyu Wang, Lei Zhang, Jianxin Liao

- **A Joint Training Dual-MRC Framework for Aspect Based Sentiment Analysis** AAAI 2021 [[paper]](https://www.aaai.org/AAAI21Papers/AAAI-5353.MaoY.pdf) 
  Yue Mao, Yi Shen, Chao Yu, LongJun Cai

- **Bidirectional Machine Reading Comprehension for Aspect Sentiment Triplet Extraction** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17500) [[arxiv]](https://arxiv.org/abs/2103.07665) [[code]](https://github.com/chenshaowei57/BMRC)  AAAI 2021 
  Shaowei Chen, Yu Wang, Jie Liu, Yuelin Wang

- **Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence.** NAACL 2019. [[paper]](https://www.aclweb.org/anthology/N19-1035/) [[code]](https://github.com/HSLCY/ABSA-BERT-pair)
  Chi Sun, Luyao Huang, Xipeng Qiu. 

- **BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis**. NAACL 2019. [[paper]](https://www.aclweb.org/anthology/N19-1242.pdf) [[code]](https://github.com/howardhsu/BERT-for-RRC-ABSA) [[dataset]](https://drive.google.com/file/d/1NGH5bqzEx6aDlYJ7O3hepZF4i_p4iMR8/view?usp=sharing)  
  Hu Xu, Bing Liu, Lei Shu and Philip S. Yu. 

- **Aspect Sentiment Classification Towards Question-Answering with Reinforced Bidirectional Attention Network.** ACL 2019. [[paper]](https://www.aclweb.org/anthology/P19-1345)
  Jingjing Wang, Changlong Sun, Shoushan Li, Xiaozhong Liu, Min Zhang, Luo Si and Guodong Zhou.


- **Document-Level Multi-Aspect Sentiment Classification as Machine Comprehension.** EMNLP 2017 [[paper]](https://aclanthology.org/D17-1217/) [[code]](https://github.com/HKUST-KnowComp/DMSC)
  Yichun Yin, Yangqiu Song, Ming Zhang


### Entity-Relation Extraction  

- **Consistent Inference for Dialogue Relation Extraction.** IJCAI 2021

- **Asking Effective and Diverse Questions: A Machine Reading Comprehension based Framework for Joint Entity-Relation Extraction.** 2020 IJCAI [[paper]](https://www.ijcai.org/proceedings/2020/546) [[code]](https://github.com/TanyaZhao/MRC4ERE_plus)
  Tianyang Zhao, Zhao Yan, Yunbo Cao, Zhoujun Li  

- **Entity-Relation Extraction as Multi-Turn Question Answering.** ACL 2019  By Jiwei Li  [[paper]](https://www.aclweb.org/anthology/P19-1129/) [[code]](https://github.com/ShannonAI/Entity-Relation-As-Multi-Turn-QA)
  Xiaoya Li, Fan Yin, Zijun Sun, Xiayu Li, Arianna Yuan, Duo Chai, Mingxin Zhou, Jiwei Li

- **Zero-Shot Relation Extraction via Reading Comprehension.** CoNLL 2017 [[paper]](https://www.aclweb.org/anthology/K17-1034/) [[code&data]](http://nlp.cs.washington.edu/zeroshot/)
  Omer Levy, Minjoon Seo, Eunsol Choi, Luke Zettlemoyer


### Event Extraction

- **What the Role Is vs. What Plays the Role: Semi-Supervised Event Argument Extraction via Dual Question Answering.** AAAI  2021  [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17720) 
  Yang Zhou, Yubo Chen, Jun Zhao, Yin Wu, Jiexin Xu, Jinlong Li. 

- **Event Extraction by Answering (Almost) Natural Questions.** EMNLP 2020. [[paper]](https://aclanthology.org/2020.emnlp-main.49/) [[code]](https://github.com/xinyadu/eeqa) 
  Xinya Du, Claire Cardie

- **Event Extraction as Multi-turn Question Answering.** EMNLP Findings 2020. [[paper]](https://aclanthology.org/2020.findings-emnlp.73/)   
Fayuan Li, Weihua Peng, Yuguang Chen, Quan Wang, Lu Pan, Yajuan Lyu, Yong Zhu
- **Event Extraction as Machine Reading Comprehension.** EMNLP 2020.  [[paper]](https://aclanthology.org/2020.emnlp-main.128/) [[code]](https://github.com/jianliu-ml/EEasMRC)
  Jian Liu ,Yubo Chen, Kang Liu, Wei Bi, Xiaojiang Liu

### NER 

- **A Unified MRC Framework for Named Entity Recognition**. 2020 ACL. [[paper]](https://aclanthology.org/2020.acl-main.519/) [[code]](https://github.com/ShannonAI/mrc-for-flat-nested-ner) [[new code]](https://github.com/wanglaiqi/mrc-for-flat-nested-ner)
Xiaoya Li, Jingrong Feng, Yuxian Meng, Qinghong Han, Fei Wu, Jiwei Li

### Summarization 

- **Improving Factual Consistency of Abstractive Summarization via Question Answering.** ACL 2021 [[paper]](https://aclanthology.org/2021.acl-long.536/)
  Feng Nan, Cicero Nogueira dos Santos, Henghui Zhu, Patrick Ng, Kathleen McKeown, Ramesh Nallapati, Dejiao Zhang, Zhiguo Wang, Andrew O. Arnold, Bing Xiang
- **QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization.** NAACL 2021 [[paper]](https://aclanthology.org/2021.naacl-main.472/) [[code]](https://github.com/Yale-LILY/QMSum)
  Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia Mutuma, Rahul Jha, Ahmed Hassan Awadallah, Asli Celikyilmaz, Yang Liu, Xipeng Qiu, Dragomir Radev

### Coreference Resolution 

- **Coreference Reasoning in Machine Reading Comprehension** ACL 2021 [[paper]](https://aclanthology.org/2021.acl-long.448/) [[code]](https://github.com/UKPLab/coref-reasoning-in-qa) 
  Mingzhu Wu, Nafise Sadat Moosavi, Dan Roth, Iryna Gurevych  

- **Ellipsis Resolution as Question Answering: An Evaluation.** EACL 2021. [[paper]](https://aclanthology.org/2021.eacl-main.68/) [[code]](https://github.com/rahular/ellipsis-baselines)
  Rahul Aralikatte, Matthew Lamm, Daniel Hardt, Anders Søgaard

- **CorefQA: Coreference Resolution as Query-based Span Prediction.**   ACL 2020 [[paper]](https://aclanthology.org/2020.acl-main.622/) [[code]](https://github.com/ShannonAI/CorefQA)
  Wei Wu, Fei Wang, Arianna Yuan, Fei Wu, Jiwei Li

- **Bridging Anaphora Resolution as Question Answering.** ACL 2020 [[paper]](https://aclanthology.org/2020.acl-main.132) [[code]](https://github.com/IBM/bridging-resolution)
  Yufang Hou  


### SQL Generation 

- **SQL Generation via Machine Reading Comprehension**. COLING 2020. [[paper]](https://aclanthology.org/2020.coling-main.31/)[[code]](https://github.com/nl2sql/QA-SQL)
  Zeyu Yan, Jianqiang Ma, Yang Zhang, Jianping Shen

### Entity Linking 

- **Read, Retrospect, Select: An MRC Framework to Short Text Entity Linking** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17528) 2021 AAAI
  Yingjie Gu, Xiaoye Qu,  Zhefeng Wang,  Baoxing Huai, Nicholas Jing Yuan,  Xiaolin Gui

### Slot Filling

- **Cross-Domain Slot Filling as Machine Reading Comprehension.**  IJCAI 2021

### Semantic Role Labeling
- **Crowdsourcing Question-Answer Meaning Representations**. NAACL 2018 [[paper]](https://aclanthology.org/N18-2089/) [[code]](https://github.com/uwnlp/qamr)
  Julian Michael, Gabriel Stanovsky, Luheng He, Ido Dagan, Luke Zettlemoyer

- **Question-Answer Driven Semantic Role Labeling: Using Natural Language to Annotate Natural Language**. EMNLP 2015 [[paper]](https://aclanthology.org/D15-1076)
  Luheng He, Mike Lewis, Luke Zettlemoyer




​


## 2. Generation-Paradigm

### NER

- **A Unified Generative Framework for Various NER Subtasks.** ACL 2021 [[paper]](https://aclanthology.org/2021.acl-long.451/) [[code]](https://github.com/yhcc/BARTNER)
  Hang Yan, Tao Gui, Junqi Dai, Qipeng Guo, Zheng Zhang, Xipeng Qiu

- **Neural Architectures for Nested NER through Linearization** ACL 2019 [[paper]](https://aclanthology.org/P19-1527/) [[code]](https://github.com/ufal/acl2019_nested_ner) 
  Jana Strakov´a, Milan Straka, and Jan Hajic.

### Aspect-Based Sentiment Analysis

- **Towards Generative Aspect-Based Sentiment Analysis.** ACL 2021 [[paper]](https://aclanthology.org/2021.acl-short.64/) [[code]](https://github.com/IsakZhang/Generative-ABSA)
  Wenxuan Zhang, Xin Li, Yang Deng, Lidong Bing, Wai Lam

- **A Unified Generative Framework for Aspect-based Sentiment Analysis.** ACL 2021 [[paper]](https://aclanthology.org/2021.acl-long.188/) [[code]](https://github.com/yhcc/BARTABSA)
  Hang Yan, Junqi Dai, Tuo Ji, Xipeng Qiu, Zheng Zhang


### Event Extraction

- **Text2Event: Controllable Sequence-to-Structure Generation for End-to-end Event Extraction.** ACL 2021  [[paper]](https://aclanthology.org/2021.acl-long.217/) [[code]](https://github.com/luyaojie/text2event)
  Yaojie Lu  Hongyu Lin  Jin Xu  Xianpei Han  Jialong Tang  Annan Li  Le Sun Meng Liao Shaoyi Chen

### Entity-Relation Extraction

- **Template Filling with Generative Transformers.**  2021 NAACL [[paper]](https://aclanthology.org/2021.naacl-main.70/) [[code]](https://github.com/xinyadu/gtt)
  Xinya Du, Alexander Rush, Claire Cardie

- **Contrastive Triple Extraction with Generative Transformer.** 2021 AAAI  [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17677) [[blog]](https://mp.weixin.qq.com/s?__biz=MzI2ODM5OTEyMA==&mid=2247484597&idx=1&sn=0a6302f88f4e891fc894d525f6e705d6&chksm=eaf163cadd86eadc172754536ea9c0f1e4b99f295fce751f0e691283fcfd21a18f1b094ae15e&mpshare=1&scene=23&srcid=0623J3VED0Rn5h3JNya7bg8o&sharer_sharetime=1624457214780&sharer_shareid=e3fac610a51f80cbddf092286e1a80a1#rd)  ZJUKG 
  Hongbin Ye, Ningyu Zhang, Shumin Deng, Mosha Chen, Chuanqi Tan, Fei Huang, Huajun Chen

- **A Unified Multi-Task Learning Framework for Joint Extraction of Entities and Relations.** AAAI 2021 [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17707)  BUAA
  Tianyang Zhao, Zhao Yan, Yunbo Cao, Zhoujun Li


- **Extracting Relational Facts by an End-to-End Neural Model with Copy Mechanism.**  ACL 2018  [[paper]](https://aclanthology.org/P18-1047/) 
  Xiangrong Zeng, Daojian Zeng, Shizhu He, Kang Liu,and Jun Zhao.



​

## 3. Span-Based-Paradigm

- **Span-based Semantic Parsing for Compositional Generalization.** ACL 2021 [[paper]](https://aclanthology.org/2021.acl-long.74/) [[code]](https://github.com/jonathanherzig/span-based-sp)
  Jonathan Herzig，Jonathan Berant

- **A Span-Based Model for Joint Overlapped and Discontinuous Named Entity Recognition.** ACL 2021 [[paper]](https://aclanthology.org/2021.acl-long.372/) [[code]](https://github.com/foxlf823/sodner) 
  Fei Li, ZhiChao Lin, Meishan Zhang, Donghong Ji

- **SpanNER: Named EntityRe-/Recognition as Span Prediction.** 2021 ACL [[arxiv]](https://arxiv.org/abs/2106.00641) [[code]](https://github.com/neulab/SpanNER) [[demo]](http://spanner.sh/) CMU
  Jinlan Fu, Xuanjing Huang, Pengfei Liu

- **Generalizing Natural Language Analysis through Span-relation Representations.** ACL 2020   [[paper]](https://aclanthology.org/2020.acl-main.192/) [[code]](https://github.com/neulab/cmu-multinlp)   CMU
  Zhengbao Jiang, Wei Xu, Jun Araki, Graham Neubig

- **Open-domain targeted sentiment analysis via span-based extraction and classification.** ACL 2019 [[paper]](https://aclanthology.org/P19-1051/) [[code]](https://github.com/huminghao16/SpanABSA)
  Minghao Hu, Yuxing Peng, Zhen Huang, Dongsheng Li, Yiwei Lv

- ​**A Span-based Joint Model for Opinion Target Extraction and Target Sentiment** ACL 2019  [[paper]](https://www.ijcai.org/proceedings/2019/0762)
  Yan Zhou, Longtao Huang, Tao Guo, Jizhong Han, Songlin Hu



​


## 4. Dependency Parsing-Paradigm

- **Structured Sentiment Analysis as Dependency Graph Parsing.** ACL 2021 [[paper]](https://aclanthology.org/2021.acl-long.263/) [[code]](https://github.com/jerbarnes/sentiment_graphs)
  Jeremy Barnes, Robin Kurtz, Stephan Oepen, Lilja Øvrelid, Erik Velldal

- **Named Entity Recognition as Dependency Parsing.** ACL 2020 [[paper]](https://aclanthology.org/2020.acl-main.577/)
  Juntao Yu, Bernd Bohnet, Massimo Poesio
 


​

## 5. Semantic Segmentation-Paradigm

- **Document-level Relation Extraction as Semantic Segmentation.** IJCAI 2021 [[arxiv]](https://arxiv.org/abs/2106.03618) [[code]](https://github.com/zjunlp/DocuNet)  [[blog]](https://mp.weixin.qq.com/s?__biz=MzI2ODM5OTEyMA==&mid=2247484958&idx=1&sn=df8c30d5ac441d6946a9e1873489899f&chksm=eaf16161dd86e8771dde37a15d2bad870fb729afe69f286fbbb7b93307f7a044a4b2d3f9b0a2&mpshare=1&scene=23&srcid=0623in4pz4nHvQqVLXQIqqHw&sharer_sharetime=1624457166808&sharer_shareid=e3fac610a51f80cbddf092286e1a80a1#rd) 
  Ningyu Zhang, Xiang Chen, Xin Xie, Shumin Deng, Chuanqi Tan, Mosha Chen, Fei Huang, Luo Si, Huajun Chen




​

## 6. Text Matching-Paradigm

- **Extractive Summarization as Text Matching**. ACL 2020 [[paper]](https://aclanthology.org/2020.acl-main.552/) [[code]](https://github.com/maszhongming/MatchSum) 
  Ming Zhong, Pengfei Liu, Yiran Chen, Danqing Wang, Xipeng Qiu, Xuanjing Huang

