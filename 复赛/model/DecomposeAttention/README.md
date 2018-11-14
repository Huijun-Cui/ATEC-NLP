这个目录下的所有文件都是Decompose Attention模型的变形。

cnn-da.py 是Decompose Attention + CNN结构

decomChar.py 是同时在char级别训练的Decompose Attention

decomWordChar 是同时在char级别和word级别训练，并且加了传统特征的Decompose Attetion

unique.py unique.py是对“独有的”字和词进行了embedding，没有传统特征的的Decompose Attetion
。

比如：句子1：“花呗还了吗？” 句子2：“花呗没还”

句子1的unique word是 “了”，“吗”  句子2的unique word是“没”

unique-cnn.py 是unique模型后面又加了cnn结构。