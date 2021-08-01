整理ing....
# 1. QA-Paradigm

- [ ] The Natural Language Decathlon: Multitask Learning as Question Answering. 2018** ** By Salesforce  [[arxiv]](https://arxiv.org/abs/1806.08730) [[code]](https://github.com/salesforce/decaNLP)​
## Aspect-Based Sentiment Analysis Task

- [x] **Question-Driven Span Labeling Model for Aspect–Opinion Pair Extraction ** AAAI 2021 ** **[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17523)

Lei Gao, Yulong Wang, Tongcun Liu, Jingyu Wang, Lei Zhang, Jianxin Liao
**概要：**采用基于Span的标注模式，将AOPE任务建模为阅读理解问题

- [x] **A Joint Training Dual-MRC Framework for Aspect Based Sentiment Analysis ** AAAI 2021 [[paper]](https://www.aaai.org/AAAI21Papers/AAAI-5353.MaoY.pdf) 

Yue Mao, Yi Shen, Chao Yu, LongJun Cai
**概要：**通过参数共享联合训练两个BERT-MRC构造两个阅读理解问题来做ASTE任务

- [x] **Bidirectional Machine Reading Comprehension for Aspect Sentiment Triplet Extraction **[[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17500)** **[[arxiv]](https://arxiv.org/abs/2103.07665) [[code]](https://github.com/chenshaowei57/BMRC)  AAAI 2021 

Shaowei Chen, Yu Wang, Jie Liu, Yuelin Wang
**概要：**将ASTE任务建模为多轮阅读理解问题，提出双向阅读理解（BMRC）框架

- [x] Chi Sun, Luyao Huang, Xipeng Qiu. **Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence. **NAACL 2019. [[paper]](https://www.aclweb.org/anthology/N19-1035/) [[code]](https://github.com/HSLCY/ABSA-BERT-pair)  

**概要：**PTMs的出现在QA、NLI等任务上取得了巨大的进步，但是在ABSA上的提升有限，作者认为是因为BERT没有利用好，作者通过构造辅助句子的形式，构造QA和NLI任务，在TABSA和ABSA任务上取得了SoTA的效果。
**亮点：**不局限于QA，还提供了NLI的query构造方式是，提供了自然句子和伪句子。
2021.7.25已阅

- [x] Hu Xu, Bing Liu, Lei Shu and Philip S. Yu. **BERT Post-Training for Review Reading Comprehension and Aspect-based Sentiment Analysis**. NAACL 2019. [[paper]](https://www.aclweb.org/anthology/N19-1242.pdf) [[code]](https://github.com/howardhsu/BERT-for-RRC-ABSA) [[dataset]](https://drive.google.com/file/d/1NGH5bqzEx6aDlYJ7O3hepZF4i_p4iMR8/view?usp=sharing)  2021.7.27阅

**概要：**构造了Review reading comprehension数据集，用阅读理解的方式做RRC任务和ABSA任务（端到端的ABSA），虽然BERT建模很有效，但是面对具体的任务相关和领域相关的，缺乏领域知识和任务相关的知识，BERT原有paper指明需要很多的数据才能完全把BERT微调好，但是一般任务并不会有很多大量的数据，因此作者使用Post-training的方式在其Amazon laptop和yelp相关的数据集上训练获得domain知识，在SQuAD上训练获得任务知识，然后进行微调，最终取得很好的效果。
**亮点：**构建了数据集，将模型传到了huggingface hub上，可以直接调用。

- [x] **Aspect Sentiment Classification Towards Question-Answering with Reinforced Bidirectional Attention Network.** ACL 2019. [[paper]](https://www.aclweb.org/anthology/P19-1345.pdf)

Jingjing Wang, Changlong Sun, Shoushan Li, Xiaozhong Liu, Min Zhang, Luo Si and Guodong Zhou.
**概要：**提出了一个新的任务，并基于淘宝标注了新的数据集。给定QA式的评论，然后做ABSC任务。介绍了当前任务的两个挑战：一个是需要同时从question和answer中抽取信息来推理情感，一个是寻找aspect相关的opinion表达时容易有不相干的词。因此引入了一种强化学习的方法来寻找aspect相关的表达，结合question和answer中的信息进行情感分类
**亮点：**提供了QA式的新数据集

- [ ] **Document-Level Multi-Aspect Sentiment Classification as Machine Comprehension.  **EMNLP 2017** **[[paper]](https://aclanthology.org/D17-1217/)** **[[code]](https://github.com/HKUST-KnowComp/DMSC)

Yichun Yin, Yangqiu Song, Ming Zhang
## Sentiment Classification

- [ ] **Sentiment classification towards question-answering with hierarchical matching network.** EMNLP 2018. [[paper]](https://aclanthology.org/D18-1401/) [[code]](https://github.com/clshenNLP/QASC)

Chenlin Shen, Changlong Sun, Jingjing Wang, Yangyang Kang, Shoushan Li, Xiaozhong Liu, Luo Si, Min Zhang, and Guodong Zhou.
## Entity-Relation Extraction Task

- [ ] **Consistent Inference for Dialogue Relation Extraction.   **IJCAI 2021
- [x] **Asking Effective and Diverse Questions: A Machine Reading Comprehension based Framework for Joint Entity-Relation Extraction. **2020 IJCAI** **[[paper]](https://www.ijcai.org/proceedings/2020/546) [[code]](https://github.com/TanyaZhao/MRC4ERE_plus) BUAA **    #worth-reading **

Tianyang Zhao, Zhao Yan, Yunbo Cao, Zhoujun Li  
**概要：**虽然已经有了用MRC做关系抽取的工作，但是依旧面临两个挑战，由于上下文语义的多样性和复杂性，构造一个问题往往很难刻画实体和关系的准确语义，同时还会导致confusing，另外现有的工作大多是枚举关系类型来构造问题，这样会导致很多的错误的问题。故此，作者设计了一个多样性QA机制来检测实体，设计了两个问题选择策略来集成不同的答案，然后预测一个潜在关系子集，进而生成问题。
**亮点：**将常见的MRC模型中预测start和end的位置改为预测BIOES标注模式
**最关键的motivation：**生成不同的问题描述使得复杂的问题更加清晰。
**具体做法：**第一，首先给每个实体类型生成各种问题，然后每个问题与context进行编码送入模型，然后选择相关的答案进行集成 最后获得头实体；第二，对于每个抽取出来的头实体进行关系预测，过滤掉一些无关的关系类型，得到一个小的关系子集；第三，尾实体抽取，给头实体，然后在子集中枚举关系进而生成问题来抽取尾实体；
**实验：**消融实验做的不错，探究了关系类型过滤、多样性问题和多个答案的集成的消融、自然问题与伪问题之间的消融、给定实体单纯做关系抽取的实验，也是比较solid

- [x] **Entity-Relation Extraction as Multi-Turn Question Answering.** ACL 2019  By Jiwei Li  [[paper]](https://www.aclweb.org/anthology/P19-1129/) [[code]](https://github.com/ShannonAI/Entity-Relation-As-Multi-Turn-QA)

Xiaoya Li, Fan Yin, Zijun Sun, Xiayu Li, Arianna Yuan, Duo Chai, Mingxin Zhou, Jiwei Li
**概要：**前人的工作，一种是pipeline的，先用序列标注识别实体，然后进行关系分类，会存在误差传递，另外一种是joint的，用多任务学习的方式，比如参数共享进行联合学习。在任务的形式上文本中蕴含的数据结构很复杂，多个实体之间还会存在依赖关系，在算法层面，现有的工作多是输入两个实体，然后进行分类，但是这样并不能很好地学到真正的语义关系。因此作者将该任务建模为阅读理解任务，构造了一个多轮的QA框架来解决这个任务。做法：构造question，首先抽取头实体（不一定都是头实体），然后枚举链式的关系模板，填入头实体、关系来构造问题，进而抽取出尾实体。将阅读理解中span start和end的预测改为 BIOE的span标注问题，4分类？，原来是N个词2分类，现在是N个词4分类？

- [ ] **Zero-Shot Relation Extraction via Reading Comprehension.** CoNLL 2017 [[paper]](https://www.aclweb.org/anthology/K17-1034/) [[code&data]](http://nlp.cs.washington.edu/zeroshot/)

Omer Levy, Minjoon Seo, Eunsol Choi, Luke Zettlemoyer
## Event Extraction

- [ ] **What the Role Is vs. What Plays the Role: Semi-Supervised Event Argument Extraction via Dual Question Answering. **AAAI  2021  [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17720) 

Yang Zhou, Yubo Chen, Jun Zhao, Yin Wu, Jiexin Xu, Jinlong Li. 
**概要：**中科院自动化所的工作，针对事件要素抽取任务，模型方面现有工作参数贡献不够完善，不利于处理稀疏数据，数据方面，前人工作大多聚焦于数据增强/生成，十分依赖外部资源。基于此，作者提出了DualQA框架将事件要素抽取任务建模为QA任务来缓解数据稀疏的问题，同时利用事件要素抽取和事件角色识别两个任务的对偶性来互相提升。 **#worth-reading **

- [ ] **Event Extraction by Answering (Almost) Natural Questions. **EMNLP ** **2020.** **[[paper]](https://aclanthology.org/2020.emnlp-main.49/) [[code]](https://github.com/xinyadu/eeqa) 
Xinya Du, Claire Cardie

概要：康奈尔大学的工作，使用QA来建模事件抽取问题，来端到端地抽取事件要素。

- [ ] **Event Extraction as Multi-turn Question Answering.** EMNLP Findings 2020. [[paper]](https://aclanthology.org/2020.findings-emnlp.73/)   
Fayuan Li, Weihua Peng, Yuguang Chen, Quan Wang, Lu Pan, Yajuan Lyu, Yong Zhu
- [ ] **Event Extraction as Machine Reading Comprehension. ** EMNLP 2020 .  [[paper]](https://aclanthology.org/2020.emnlp-main.128/) [[code]](https://github.com/jianliu-ml/EEasMRC)
Jian Liu ,Yubo Chen, Kang Liu, Wei Bi, Xiaojiang Liu

**概要：**中科院自动化所的工作，将事件抽取任务建模为阅读理解问题。
## NER Task

- [x] **A Unified MRC Framework for Named Entity Recognition**. 2020 ACL. [[paper]](https://aclanthology.org/2020.acl-main.519/) [[code]](https://github.com/ShannonAI/mrc-for-flat-nested-ner) [**[new code]**](https://github.com/wanglaiqi/mrc-for-flat-nested-ner)
Xiaoya Li, Jingrong Feng, Yuxian Meng, Qinghong Han, Fei Wu, Jiwei Li

**概要：**早期工作主要是将其建模为序列标注问题，但是这样只能给每个token分配一个标签，不能很好地处理overlap或者nested的问题，针对nested-NER很多pipeline的工作，但是会涉及到误差传递的问题。使用MRC建模NER任务，十分优雅地融入了先验知识，能够对nested-NER或者overlap-NER进行处理，统一了flat-NER和nested-NER两个子任务，并且在nested相关的数据集上获得了很大的提升。亮点：除了想法新颖之外，实验也做的很solid，比如消融实验，编码器与MRC模型的消融、query的构造消融、zero-shot的评估、训练数据验证，整体上来讲十分solid。此外还在start和end的index匹配上额外加了一点小trick。
## Summarization Task

- [ ] **Improving Factual Consistency of Abstractive Summarization via Question Answering. **ACL 2021 [[paper]](https://aclanthology.org/2021.acl-long.536/) 
Feng Nan, Cicero Nogueira dos Santos, Henghui Zhu, Patrick Ng, Kathleen McKeown, Ramesh Nallapati, Dejiao Zhang, Zhiguo Wang, Andrew O. Arnold, Bing Xiang
- [ ] **QMSum: A New Benchmark for Query-based Multi-domain Meeting Summarization. ** NAACL** **2021 [[paper]](https://aclanthology.org/2021.naacl-main.472/) [[code]](https://github.com/Yale-LILY/QMSum)

Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia Mutuma, Rahul Jha, Ahmed Hassan Awadallah, Asli Celikyilmaz, Yang Liu, Xipeng Qiu, Dragomir Radev
## Coreference Resolution Task

- [x] **Coreference Reasoning in Machine Reading Comprehension **[[paper]](https://aclanthology.org/2021.acl-long.448/) [[code]](https://github.com/UKPLab/coref-reasoning-in-qa) ACL 2021

Mingzhu Wu, Nafise Sadat Moosavi, Dan Roth, Iryna Gurevych  
**概要：**针对阅读理解中的指代推理问题，提出了一种将指代推理转化为MRC格式数据集的方法，作者认为现有的方法虽然取得了很好的效果，但是数据集是有偏差的，并不能真实地反应指代问题，因此作者标注了一个小数据量的challenge dataset（QA）范式的，然后做了一些实验，证明该领域还有很大的空间。
**亮点：**这里面的问题是使用BART生成的。

- [ ] **Ellipsis Resolution as Question Answering: An Evaluation.** EACL 2021. [[paper]](https://aclanthology.org/2021.eacl-main.68/) [[code]](https://github.com/rahular/ellipsis-baselines)

Rahul Aralikatte, Matthew Lamm, Daniel Hardt, Anders Søgaard

- [ ] **CorefQA: Coreference Resolution as Query-based Span Prediction.**   ACL 2020 [[paper]](https://aclanthology.org/2020.acl-main.622/) [[code]](https://github.com/ShannonAI/CorefQA)
Wei Wu, Fei Wang, Arianna Yuan, Fei Wu, Jiwei Li
- [ ] **Bridging Anaphora Resolution as Question Answering. **ACL 2020** **[[paper]](https://aclanthology.org/2020.acl-main.132) [[code]](https://github.com/IBM/bridging-resolution)

 Yufang Hou  
## SQL Generation Task

- [ ] **SQL Generation via Machine Reading Comprehension**. COLING 2020. [[paper]](https://aclanthology.org/2020.coling-main.31/)** **[[code]](https://github.com/nl2sql/QA-SQL) 
Zeyu Yan, Jianqiang Ma, Yang Zhang, Jianping Shen

概要：平安团队的工作，将SQL生成任务建模为MRC问题。 （代码涉及隐私问题尚未开源）
## Entity Linking Task

- [ ] **Read, Retrospect, Select: An MRC Framework to Short Text Entity Linking** [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17528) 2021 AAAI

Yingjie Gu, Xiaoye Qu,  Zhefeng Wang,  Baoxing Huai, Nicholas Jing Yuan,  Xiaolin Gui
## Slot Filling

- [ ] **Cross-Domain Slot Filling as Machine Reading Comprehension. **IJCAI 2021



# 2. Dialogue-Paradigm

- [ ] **Dialogue-Based Relation Extraction**. ACL 2020.  [[paper]](https://aclanthology.org/2020.acl-main.444/) [[data]](https://dataset.org/dialogre/)

Dian Yu, Kai Sun, Claire Cardie, Dong Yu. 
**概要：**腾讯和康奈尔的工作，作者提出了第一个人工标注的基于对话的关系抽取数据集来支持对话中两个要素的关系预测，并且进一步将DialogRE作为一个用于跨句子关系抽取的平台，并且在这种设定下设计了新的评估指标。
# 3. Generation-Paradigm

- [ ] **A Unified Generative Framework for Various NER Subtasks. **ACL 2021 [[paper]](https://aclanthology.org/2021.acl-long.451/) [[code]](https://github.com/yhcc/BARTNER)

Hang Yan, Tao Gui, Junqi Dai, Qipeng Guo, Zheng Zhang, Xipeng Qiu

- [ ] **Towards Generative Aspect-Based Sentiment Analysis. **ACL 2021 [[paper]](https://aclanthology.org/2021.acl-short.64/) [[code]](https://github.com/IsakZhang/Generative-ABSA)

Wenxuan Zhang, Xin Li, Yang Deng, Lidong Bing, Wai Lam

- [ ] **A Unified Generative Framework for Aspect-based Sentiment Analysis. **ACL 2021 [[paper]](https://aclanthology.org/2021.acl-long.188/) [[code]](https://github.com/yhcc/BARTABSA)

Hang Yan, Junqi Dai, Tuo Ji, Xipeng Qiu, Zheng Zhang

- [ ] **Text2Event: Controllable Sequence-to-Structure Generation for End-to-end Event Extraction.** ACL 2021  [[paper]](https://aclanthology.org/2021.acl-long.217/) [[code]](https://github.com/luyaojie/text2event)
Yaojie Lu  Hongyu Lin  Jin Xu  Xianpei Han  Jialong Tang  Annan Li  Le Sun Meng Liao Shaoyi Chen
- [ ] **Template Filling with Generative Transformers.**  2021 NAACL [[paper]](https://aclanthology.org/2021.naacl-main.70/) [[code]](https://github.com/xinyadu/gtt)

Xinya Du, Alexander Rush, Claire Cardie
**概要：**康奈尔大学的工作，使用端到端的生成transformer来对模板填充（Template Filling）任务建模，比pipeline的方法性能更佳。

- [ ] **Contrastive Triple Extraction with Generative Transformer. **2021 AAAI  [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17677) [[blog]](https://mp.weixin.qq.com/s?__biz=MzI2ODM5OTEyMA==&mid=2247484597&idx=1&sn=0a6302f88f4e891fc894d525f6e705d6&chksm=eaf163cadd86eadc172754536ea9c0f1e4b99f295fce751f0e691283fcfd21a18f1b094ae15e&mpshare=1&scene=23&srcid=0623J3VED0Rn5h3JNya7bg8o&sharer_sharetime=1624457214780&sharer_shareid=e3fac610a51f80cbddf092286e1a80a1#rd)  ZJUKG 

Hongbin Ye, Ningyu Zhang, Shumin Deng, Mosha Chen, Chuanqi Tan, Fei Huang, Huajun Chen

- [ ] **A Unified Multi-Task Learning Framework for Joint Extraction of Entities and Relations. **AAAI 2021 [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17707)  BUAA
- [ ] **Don’t Eclipse Your Arts Due to Small Discrepancies: Boundary Repositioning with a Pointer Network for Aspect Extraction**. ACL 2020. [[paper]](https://www.aclweb.org/anthology/2020.acl-main.339.pdf) [[code]](https://www.aclweb.org/anthology/attachments/2020.acl-main.339.Software.zip)

Zhenkai Wei, Yu Hong, Bowei Zou, Meng Cheng, Jianmin YAO. 

- [ ] **Conditional Augmentation for Aspect Term Extraction via Masked Sequence-to-Sequence Generation**. ACL 2020. [[paper]](https://www.aclweb.org/anthology/2020.acl-main.631.pdf)

Kun Li, Chengbo Chen, Xiaojun Quan, Qing Ling, Yan Song. 

- [ ] **Extracting Relational Facts by an End-to-End Neural Model with Copy Mechanism.**  ACL 2018  [[paper]](https://aclanthology.org/P18-1047/) 

Xiangrong Zeng, Daojian Zeng, Shizhu He, Kang Liu,and Jun Zhao.

- [ ] **Neural Architectures for Nested NER through Linearization **[[paper]](https://aclanthology.org/P19-1527/) [[code]](https://github.com/ufal/acl2019_nested_ner) ACL 2019

Jana Strakov´a, Milan Straka, and Jan Hajic.
概要：将NER任务建模为seq2seq的范式
​

# 4. Span-Based-Paradigm

- [ ] **Span-based Semantic Parsing for Compositional Generalization. **ACL 2021** **[[paper]](https://aclanthology.org/2021.acl-long.74/) [[code]](https://github.com/jonathanherzig/span-based-sp)
Jonathan Herzig，Jonathan Berant
- [ ] **A Span-Based Model for Joint Overlapped and Discontinuous Named Entity Recognition.** ACL 2021 [[paper]](https://aclanthology.org/2021.acl-long.372/) [[code]](https://github.com/foxlf823/sodner)  **#worth-reading  解码算法！**

Fei Li, ZhiChao Lin, Meishan Zhang, Donghong Ji

- [ ] **SpanNER: Named EntityRe-/Recognition as Span Prediction. **2021 ACL [[arxiv]](https://arxiv.org/abs/2106.00641) [[code]](https://github.com/neulab/SpanNER) [[demo]](http://spanner.sh/) CMU

Jinlan Fu, Xuanjing Huang, Pengfei Liu
**概要：**NER任务的范式转移，试图在现有的将序列标注转为Span prediction的工作基础上探寻现有span-based模型的不足之处，对不同的model做了ensemble，偏工程的文章。

- [ ] **Generalizing Natural Language Analysis through Span-relation Representations.** 2020 ACL  [[paper]](https://aclanthology.org/2020.acl-main.192/) [[code]](https://github.com/neulab/cmu-multinlp)   CMU

Zhengbao Jiang, Wei Xu, Jun Araki, Graham Neubig
**概要：**作者试图将NLP任务统一成标记span和关系的范式，在10种不同的NLP任务上进行了实验，并且将对应的数据集构造成立新的benchmark。

- [x] **Open-domain targeted sentiment analysis via span-based extraction and classification. **2019 ACL [[paper]](https://aclanthology.org/P19-1051/) [[code]](https://github.com/huminghao16/SpanABSA)  2021.7.24已阅

Minghao Hu, Yuxing Peng, Zhen Huang, Dongsheng Li, Yiwei Lv
**概要：**针对aspect-sentiment co-extraction任务，提出了一个基于span boundary prediction的模型，先抽取aspect，然后情感极性分类，超过了现有的SOTA。与此同时还提出了multi-target extractor和对应的启发式解码算法。
**亮点：**multi-target extractor，启发式的解码算法，不过这个解码算法依赖阈值的设定，或许可以做进一步改进；实验方面：消融时针对不同长度的aspect进行实验，检测基于SPAN和TAG的效果在不同长度时哪个更好。
Insight：SPAN-based的模型在短文本上处理的并不好，有可能是解码算法的问题，文章声称说pipeline的模型比joint的好，其实pipeline的模型用了两个BERT，一个做aspect抽取，一个做分类，这样自然比joint的好，orz..
**瑕疵：**实验方面少了一个BERT/LSTM的消融

- [ ] ​**A Span-based Joint Model for Opinion Target Extraction and Target Sentiment**   2019 ACL [[paper]](https://www.ijcai.org/proceedings/2019/0762)

Yan Zhou, Longtao Huang, Tao Guo, Jizhong Han, Songlin Hu


# 5. Dependency Parsing-Paradigm

- [ ] **Structured Sentiment Analysis as Dependency Graph Parsing. ACL 2021 **[[paper]](https://aclanthology.org/2021.acl-long.263/)** **[[code]](https://github.com/jerbarnes/sentiment_graphs)

Jeremy Barnes, Robin Kurtz, Stephan Oepen, Lilja Øvrelid, Erik Velldal

- [ ] **Named Entity Recognition as Dependency Parsing.** ACL 2020 [[paper]](https://aclanthology.org/2020.acl-main.577/)
Juntao Yu, Bernd Bohnet, Massimo Poesio 
# 6. Semantic Segmentation-Paradigm

- [ ] **Document-level Relation Extraction as Semantic Segmentation.** IJCAI 2021 [[arxiv]](https://arxiv.org/abs/2106.03618) [[code]](https://github.com/zjunlp/DocuNet)  [[blog]](https://mp.weixin.qq.com/s?__biz=MzI2ODM5OTEyMA==&mid=2247484958&idx=1&sn=df8c30d5ac441d6946a9e1873489899f&chksm=eaf16161dd86e8771dde37a15d2bad870fb729afe69f286fbbb7b93307f7a044a4b2d3f9b0a2&mpshare=1&scene=23&srcid=0623in4pz4nHvQqVLXQIqqHw&sharer_sharetime=1624457166808&sharer_shareid=e3fac610a51f80cbddf092286e1a80a1#rd) 

Ningyu Zhang, Xiang Chen, Xin Xie, Shumin Deng, Chuanqi Tan, Mosha Chen, Fei Huang, Luo Si, Huajun Chen


# 7. Text Matching-Paradigm

- [ ] **Extractive Summarization as Text Matching **ACL 2020_  _[[paper]](Extractive Summarization as Text Matching) [[code]](https://github.com/maszhongming/MatchSum) 

Ming Zhong*, Pengfei Liu*, Yiran Chen, Danqing Wang, Xipeng Qiu, Xuanjing Huang

- [ ] [**Label Correction Model for Aspect-based Sentiment Analysis**](https://aclanthology.org/2020.coling-main.71/)**     2020COLING[**Qianlong Wang**](https://aclanthology.org/people/q/qianlong-wang/)** | **[**Jiangtao Ren**](https://aclanthology.org/people/j/jiangtao-ren/)





先驱工作：

- [ ] The Natural Language Decathlon: Multitask Learning as Question Answering. **2018 ** By Salesforce  [[arxiv]](https://arxiv.org/abs/1806.08730) [[code]](https://github.com/salesforce/decaNLP)  leaderboard
- [ ] Zero-Shot Relation Extraction via Reading Comprehension. CoNLL 2017 By Washington NLP  [[paper]](https://www.aclweb.org/anthology/K17-1034/) [[code&data]](http://nlp.cs.washington.edu/zeroshot/)



​

