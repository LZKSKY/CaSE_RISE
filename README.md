## Process

#### 1. Create Dataset with Label

1. Run Preprocess/prepare_mld.py

2. Use utils_.py : TagSeq

   merge path

   ​	merge: multiple substitute

   ​	merge: multiple Insert

   ​	add <BOS> token

   finally, it will output len([<bos>] + input_seq) labels



#### 2. Write Dataset

1. In dataset/MLDDataset.py
2. it contains
   1. $c$, $x$ = $V^{190}, V^{32}$
   2. $x_{pos}, x_{s}$: pos id, segment id
   3. edit label: $l$  operation in ['K', 'S', 'I', 'D']
3. for substitute and insert, we use a decoder to generate
   1. the max decoder_len = 20
   2. max [insert, substitute] = 5
   3. 最多允许5次插入，每次插入长度不超过20
   4. 对于 insert， $x_i, x_j$之间插入的时候，$q = [x_i, [SEP], x_j]$
   5. 对于 substitute, 替换$x_i, ...,x_j$的时候，$q=[x_i, ...,x_j]$

#### 3. Write model

1. Model/BertMLD.py
   1. forward
   2. generate
2. Because it can not generate his, he, here, we address it as a classification task





---

#### ~~4. Predict model rather generate~~

1. ~~build vocab~~
   1. ~~insert phrase and all insert tokens~~
2. ~~define reward~~
   1. ~~我们要减去baseline，不如，对于每个sample，每次存一个b~~
   2. 



#### 4. Generate path

1. implement a DP path generated methods
2. implement a action probability methods
3. implement a final path generation methods