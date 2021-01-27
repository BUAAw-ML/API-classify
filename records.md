

# 实验结果 -1205

## programmerWeb数据集

### tag频率<200的数据集

|实验环境|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|---|
|614服务器(conda:discrete)|L,U,T:3645,0,1563(split:0.7,未加title_ids)|Bert微调+多注意力|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.01,B0.1|64.618|
|苏州服务器|L,U,T:3645,0,1563(split:0.7,未加title_ids)|Bert微调+多注意力|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.01,B0.1|65.03|
|614服务器(conda:discrete)|L,U,T:2604,0,1563(split:0.5,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|62.620|
|614服务器(conda:discrete)|L,U,T:1562,0,1563(split:0.3,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|58.845|
|苏州服务器|L,U,T:260,,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|38.292  37.583  38.411  38.042  37.163  35.323  34.915  37.344|
|苏州服务器|L,U,T:260,,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|40.888  39.399  37.532|
|---|---|---|---|---|
|614服务器(conda:discrete)|L,U,T:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和);D训练未使用G处理后的噪音数据|epoch:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.369  46.496|
|614服务器(conda:discrete)|L,U,T:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和);D训练未使用G处理后的噪音数据|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.315  47.826  47.296|
|苏州服务器|L,U,T:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|51.008  51.994  49.915  51.133|
|苏州服务器|L,U,T:520,3125,1563(tsplit:0.1,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|54.454|
|苏州服务器|L,U,T:1040,2605,1563(split:0.2,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|58.86|
|614服务器(conda:discrete)|L,U,T:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|50.9  47.756|
|614服务器(conda:discrete)|L,U,T:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|51.422  51.364 51.543|
|苏州服务器|L,U,T:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN；Generator设置2层（没提的都为1层）|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|52.123|


另外，进行了其它试验，包括：
- 模型在Gnerator学习率为0.0001，0.01下效果不好;
- Gnerator用了G_feat_match效果不好；
- Gnerator设置3层不好，使用dropout不好；
- 只用label做生成对抗反而不好了

### 全部数据集

|实验环境|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|---|
|苏州服务器|L,U,T:612,0,1226(split:0.05,0.9,未加title_ids)|Bert微调+多注意力|epoch:50;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|40.378|
|苏州服务器|L,U,T:8579,0,1226(split:0.7,0.9,未加title_ids)|Bert微调+多注意力|epoch:50;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|59.774|
|苏州服务器|L,U,T:11030,0,1226(split:0.9,未加title_ids)|Bert微调+多注意力|epoch:50;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|62|
|苏州服务器|L,U,T:612,,1226(split:0.05,0.9,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;epoch_step:45；batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|44|
|苏州服务器|L,U,T:2448,,1226(split:0.2,0.9,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;epoch_step:45；batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|51.098|

小结：
- 采用全部数据集（115个标签）时，提出的方法的效果只好大概百分之四，不是很明显；
- programmerWeb数据集性上模型训练到d_loss变为0的时候性能不会下降，还会略微慢慢提升；
- 去掉标注数据的无监督损失不影响最终性能 

### tag频率<100的数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:72,0,438(split:0.05,0.7,未加title_ids)|Bert微调+多注意力|epoch:20;epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|46.170  45.024  48.917  46.872  43.485  47.755|
|L,U,T:72,949,438(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|54.745  54.383  55.709  52.864|

另外，进行了其它试验，包括：
- model里的判别特征如何改成和权重矩阵乘后求mean()效果是不好的。
- 尝试了对所有未标注样本打伪标签（预测概率最大的类别tag设置为1），然后一起训练模型。但是基本训练不起来。

## gan-bert数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:109,0,500|Bert微调+多注意力|epoch:20;epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|20  22  23|
|L,U,T:109,5343,500|Bert微调+多注意力+GAN|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|28|


## AAPD数据集
标签数：54
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:548,,3291|Bert微调+多注意力|epoch:50;epoch_step:15;batch-size:30;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|28.599  28|
|L,U,T:43323,,10968|Bert微调+多注意力|epoch:8;epoch_step:;batch-size:30;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|58.760|
|L,U,T:49301,,5484|Bert微调+多注意力|epoch:;epoch_step:;batch-size:30;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|60.641|
|---|---|---|---|
|L,U,T:548,,16452|Bert微调+多注意力|epoch:21;epoch_step:15;batch-size:30;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|28.793|
|L,U,T:548,37840,16452|Bert微调+多注意力+GAN|epoch:10;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|34.139|
|L,U,T:548,37840,16452|Bert微调+多注意力+GAN|epoch:15;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|31.651|
|---|---|---|---|
|L,U,T:548,,3291|Bert微调+多注意力|epoch:50;epoch_step:15;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|34.768|
|L,U,T:548,7129,3291|Bert微调+多注意力+GAN|epoch:23;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|39.414|
|---|---|---|---|
|L,U,T:548,,3291|Bert微调+多注意力|epoch:50;epoch_step:15;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|34.129  32.716|
|L,U,T:548,7129,3291|Bert微调+多注意力+GAN|epoch:23;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|40.419|
|---|---|---|---|
|L,U,T:4387,,2194|Bert微调+多注意力|epoch:50;epoch_step:15;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|48.924  48.842|
|L,U,T:4387,4387,2194|Bert微调+多注意力+GAN|epoch:20;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|50.165|
|---|---|---|---|
|L,U,T:7677,,2194|Bert微调+多注意力|epoch:31;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|52.405  51.908|
|L,U,T:7677,1097,2194|Bert微调+多注意力+GAN|epoch:;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|53.727|

另外，进行了其它试验，包括：
- 0.69的label，0.01的。
- batch-size使用30时，GAN初期提不起来（6轮都不咋提高），感觉之后效果应该不好。
- 提出方法当模型达到最高性能后性能又会快速下降（掉到底）（好像是在d_loss变为0的时候）
- 感觉batch-size对方法的效果有影响
- 给generator增加了一层也还是不能避免d_loss变为0后性能迅速下降

## EUR-Lex数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:2176,,1084（标签数：60）（tag频率>200,text_len<510）|Bert微调+多注意力|epoch:22;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|55.069|
|L,U,T:2176,2177,1084（标签数：60）（tag频率>200,text_len<510）|Bert微调+多注意力+GAN|epoch:45;epoch_step:40;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|55.577|
|L,U,T:4353,,1084（标签数：60）（tag频率>200,text_len<510）|Bert微调+多注意力|epoch:45;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|62.487|
|---|---|---|---|
|L,U,T:5422,,2662（标签数：60）（tag频率>200,text按510截断）|Bert微调+多注意力+GAN|epoch:23;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|63.559|
|L,U,T:5422,5423,2662（标签数：60）（tag频率>200,text按510截断）|Bert微调+多注意力+GAN|epoch:35;epoch_step:40;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|63.904|
|L,U,T:10845,,2662（标签数：60）（tag频率>200,text按510截断）|Bert微调+多注意力|epoch:17;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|66.914|
|---|---|---|---|
|L,U,T:870,,1084（标签数：60）（tag频率>200,skip）|Bert微调+多注意力+GAN|epoch:30;epoch_step:15;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|45.705|
|L,U,T:870,3483,1084（标签数：60）（tag频率>200,skip）|Bert微调+多注意力+GAN|epoch:60;epoch_step:50;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.850|
|---|---|---|---|
|L,U,T:435，，1084（标签数：60）（tag频率>200,skip）|Bert微调+多注意力+GAN|epoch:30;epoch_step:15;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|37.547|
|L,U,T:435，3918，1084（标签数：60）（tag频率>200,skip）|Bert微调+多注意力+GAN|epoch:60;epoch_step:50;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|37.876|
|---|---|---|---|


另外进行的试验：
- 过滤文本长于510，且使用标签频次大于100  能达到五十多的MAP
- 使用标签频次大于100 能达到三十多的MAP
- 使用标签频次大于10 有一千三百八十多个标签 用一半的训练数据 截断的能达到十七的MAP 过滤的（不截断）的七点多的MAP
- 63.452 训练集和测试集都是相同的一百八十多个标签
- 在tag频率>200（190个tag）,skip时（训练了50轮和80轮），提出方法效果在split=0.05,0.1,0.2时都较差，其在labeled数据集训练精读也上不到最高，而且好像此时d_Loss都为0。

## RCV2数据集
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:800,,132（标签数：41）（tag频率<5000,intanceNum_limit<5000）|Bert微调+多注意力|epoch:20;epoch_step:15;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|55.890|
|L,U,T:800,3201,132（标签数：41）（tag频率<5000,intanceNum_limit<5000）|Bert微调+多注意力+GAN|epoch:40;epoch_step:30;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|58.535|
|---|---|---|---|
|L,U,T:1000,,410（标签数：41）（tag频率<20000,intanceNum_limit<10000）|Bert微调+多注意力|epoch:25;epoch_step:15;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|44.963|
|L,U,T:1000,9001,410（标签数：41）（tag频率<20000,intanceNum_limit<10000）|Bert微调+多注意力+GAN|epoch:40;epoch_step:30;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.398|

另外进行的试验：
- 提出方法当模型达到最高性能后性能又会快速下降（好像是在d_loss变为0的时候）
- 使用该数据1500，13501，668 提出的方法没有训练成果，具体因为训练中性能掉到底两次

## Stack Overflow数据集
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:500,,262（标签数：205）（tag频率<200,intanceNum_limit<5000）|Bert微调+多注意力|epoch:30;epoch_step:20;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|34.023|
|L,U,T:500,4501,262（标签数：205）（tag频率<200,intanceNum_limit<5000）|Bert微调+多注意力+GAN|epoch:75;epoch_step:65;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|40.081|


另外进行的试验：
- 采用大规模数据集时，例如tag频率<500，提出方法性能提升到一半就提不动了。
- 在上面第二个实验中，即使当D_loss变为0，模型性能依然能提升。
- 在上面第二个实验中，使用提出方法的修改（fine-grained），性能超过了baseline，但是忽然D_loss变为0，模型性能提升就停滞了
- 学习率、批大小、优化器（Adam训练不起来）都无法解决不能进一步提高的问题
- 把标注数据放在前面训练效果好一点

# 实验结果 1205-1226
## programmerWeb数据集

实验发现：
- 数据量较少时，多注意力后用一个线性层效果好；数据量较多时，多注意力后用分别的权重效果好。
- 采用tag注意力其实就是：计算出每个tag的注意力后，其中最大注意力值若在所有tag里还最大，则该tag很大可能就为预测结果。
- 随着训练进行，tag和token的平均similarity会越来越小（从正到负）
- 把对于真实类别的类别预测概率平均值通过设置loss函数快速抬起来往往导致训练崩溃，因为所有类别的预测概率都被一起抬起来了。

# 实验结果 1205-1226
## agnews数据集
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:12000,,7600（标签数：4）（split:0.1）|Bert微调+多注意力|epoch:13;epoch_step:8;batch-size:4;optimizer:SGD;learning-rate:G0.0001,D0.01,B0.0001|96.494|


# 实验结果 0104-0107
## agnews数据集
用sigmoid，不用unlabel数据。
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:40,,7600（标签数：4）|Bert微调+多注意力|epoch:70;epoch_step:60;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.01,B0.0001|75.866|
|L,U,T:40,,7600（标签数：4）|Bert微调+多注意力|epoch:70;epoch_step:60;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.01,B0.001|84.424|
|L,U,T:40,500,7600（标签数：4）|Bert微调+多注意力+GAN|epoch:70;epoch_step:60;batch-size:4;optimizer:SGD;learning-rate:G0.0001,D0.01,B0.0001|83.882|
|L,U,T:40,500,7600（标签数：4）|Bert微调+多注意力+GAN|epoch:70;epoch_step:60;batch-size:4;optimizer:SGD;learning-rate:G0.0001,D0.01,B0.001|86.322|
|---|---|---|---|
|L,U,T:800,,7600（标签数：4）|Bert微调+多注意力|epoch:70;epoch_step:60;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.01,B0.0001|88.391|
|L,U,T:800,500,7600（标签数：4）|Bert微调+多注意力+GAN|epoch:70;epoch_step:60;batch-size:4;optimizer:SGD;learning-rate:G0.0001,D0.01,B0.0001|87.271| 
|L,U,T:40,500,7600（标签数：4）|Bert微调+多注意力+GAN|epoch:70;epoch_step:60;batch-size:4;optimizer:SGD;learning-rate:G0.0001,D0.01,B0.001|87.671|
|---|---|---|---|
|L,U,T:10000,,7600（标签数：4）|Bert微调+多注意力|epoch:16;epoch_step:60;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.01,B0.0001|95.605|
|L,U,T:10000,500,7600（标签数：4）|Bert微调+多注意力+GAN|epoch:20;epoch_step:60;batch-size:4;optimizer:SGD;learning-rate:G0.0001,D0.01,B0.0001|95.504|
|L,U,T:10000,500,7600（标签数：4）|Bert微调+多注意力+GAN|epoch:20;epoch_step:60;batch-size:4;optimizer:SGD;learning-rate:G0.0001,D0.01,B0.0001|95.923|
|---|---|---|---|

方法小结：
- 生成对抗和一致性训练是两种不同的半监督学习方法，都是利用未标注数据集的。

# 实验结果 0109
## Stack Overflow数据集
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,701（标签数：97）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|33.823|
|L,U,T:200,400,701（标签数：97）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|40.782|
|L,U,T:400,,701（标签数：97）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|50.345|
|L,U,T:400,400,701（标签数：97）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|54.779|
|L,U,T:1600,,701（标签数：97）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|65.805|
|L,U,T:1600,400,701（标签数：97）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|65.959|
|L,U,T:6400,,701（标签数：97）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|72.454|
|L,U,T:6400,400,701（标签数：97）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|72.410|
|---|---|---|---|

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,984（标签数：192）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|24.643|
|L,U,T:200,400,984（标签数：192）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|30.427|
|L,U,T:400,,984（标签数：192）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|32.860|
|L,U,T:400,400,984（标签数：192）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|43.728|
|L,U,T:1600,,984（标签数：192）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|57.849|
|L,U,T:1600,400,984（标签数：192）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|57.233|
|L,U,T:6400,,984（标签数：192）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|65.727|
|L,U,T:6400,400,984（标签数：192）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|66.126|
|---|---|---|---|

# 实验结果 0110
## Stack Overflow数据集
把有'#'的标签的'#'去除，原来四百三十多的标签合并为了三百多个

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,（标签数：100）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=40.648 OP=0.297 OR=0.283 OF1=0.402 CP=0.521 CR=0.249 CF1=0.337|
|L,U,T:200,400,（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001| map=40.482 OP=0.274 OR=0.209 OF1=0.322 CP=0.438 CR=0.168 CF1=0.243|
|L,U,T:400,,（标签数：100）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=49.553 OP=0.389 OR=0.372 OF1=0.501 CP=0.635 CR=0.353 CF1=0.453|
|L,U,T:400,400,（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=52.773 OP=0.419 OR=0.373 OF1=0.503 CP=0.636 CR=0.357 CF1=0.457|
|L,U,T:1600,,（标签数：100）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=62.693 OP=0.494 OR=0.586 OF1=0.652 CP=0.721 CR=0.577 CF1=0.641|
|L,U,T:1600,400,（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=63.281 OP=0.484 OR=0.595 OF1=0.651 CP=0.732 CR=0.595 CF1=0.656|
|L,U,T:6400,,（标签数：100）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=68.255 OP=0.527 OR=0.645 OF1=0.687 CP=0.745 CR=0.637 CF1=0.687|
|L,U,T:6400,400,（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=69.144 OP=0.528 OR=0.625 OF1=0.689 CP=0.773 CR=0.621 CF1=0.689|
|L,U,T:12800,,（标签数：100）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:12800,400,（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,（标签数：200）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=30.982 OP=0.200 OR=0.168 OF1=0.266 CP=0.322 CR=0.142 CF1=0.197|
|L,U,T:200,400,（标签数：200）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=33.496 OP=0.163 OR=0.102 OF1=0.177 CP=0.235 CR=0.082 CF1=0.121|
|L,U,T:400,,（标签数：200）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=41.521 OP=0.282 OR=0.289 OF1=0.409 CP=0.462 CR=0.259 CF1=0.332|
|L,U,T:400,400,（标签数：200）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=46.849 OP=0.326 OR=0.301 OF1=0.426 CP=0.470 CR=0.261 CF1=0.335|
|L,U,T:1600,,（标签数：200）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=56.009 OP=0.392 OR=0.522 OF1=0.594 CP=0.636 CR=0.512 CF1=0.567|
|L,U,T:1600,400,（标签数：200）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=55.487 OP=0.390 OR=0.527 OF1=0.588 CP=0.613 CR=0.523 CF1=0.565|
|L,U,T:6400,,（标签数：200）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=61.044 OP=0.418 OR=0.577 OF1=0.632 CP=0.669 CR=0.562 CF1=0.611|
|L,U,T:6400,400,（标签数：200）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=60.946 OP=0.406 OR=0.555 OF1=0.629 CP=0.656 CR=0.549 CF1=0.598|
|L,U,T:12800,,（标签数：200）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:12800,400,（标签数：200）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

数据按类别分割、打乱，GAN（不用未标注样本）方法试验（0<tag<100_RCV2、0<tag_AAPD、10<tag<60_EUR-Lex、60<tag<100_RCV2）：
-在RCV2数据集没有效果
-在AAPD数据集没有效果
-在EUR-Lex数据集没有效果

# 实验结果 0111-0112
label和unlabel分开训练，并加上无监督损失D_L_unsupervised2

##AAPD数据集
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,12398（标签数：54）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=26.338,25.705 OP=0.000,0.001 OR=0.369,0.355 OF1=0.490,0.477 CP=0.427,0.415 CR=0.158,0.149 CF1=0.231,0.219|
|L,U,T:200,400,12398（标签数：54）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=29.934 OP=0.000 OR=0.392 OF1=0.505 CP=0.373 CR=0.179 CF1=0.242|
|L,U,T:400,,12398（标签数：54）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=31.470,32.582 OP=0.002,0.008 OR=0.430,0.452 OF1=0.544,0.553 CP=0.465,0.448 CR=0.206,0.242 CF1=0.286,0.314|
|L,U,T:400,400,12398（标签数：54）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=36.022 OP=0.001 OR=0.475 OF1=0.567 CP=0.494 CR=0.250 CF1=0.332|
|L,U,T:1600,,12398（标签数：54）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=42.951, OP=0.002, OR=0.554, OF1=0.624, CP=0.551, CR=0.357, CF1=0.433,|
|L,U,T:1600,400,12398（标签数：54）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=43.622 OP=0.040 OR=0.575 OF1=0.622 CP=0.517 CR=0.363 CF1=0.427|
|L,U,T:6400,,12398（标签数：54）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=49.288,48.740 OP=0.000,0.003 OR=0.614,0.604 OF1=0.654,0.654 CP=0.550,0.554 CR=0.435,0.426 CF1=0.486,0.481|
|L,U,T:6400,400,12398（标签数：54）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|
|L,U,T:200,800,12398（标签数：54）|Bert微调+多注意力GAN3|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=29.406 OP=0.000 OR=0.361 OF1=0.483 CP=0.427 CR=0.151 CF1=0.223|
|L,U,T:400,800,12398（标签数：54）|Bert微调+多注意力GAN3|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=35.365 OP=0.000 OR=0.458 OF1=0.561 CP=0.501 CR=0.236 CF1=0.320|
|L,U,T:1600,800,12398（标签数：54）|Bert微调+多注意力GAN3|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:6400,800,12398（标签数：54）|Bert微调+多注意力GAN3|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

##RCV2数据集
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 200,,1191（标签数：103）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=20.845 OP=0.016 OR=0.458 OF1=0.565 CP=0.230 CR=0.130 CF1=0.166|
|L,U,T: 400,,1191（标签数：103）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=28.924 OP=0.014 OR=0.561 OF1=0.648 CP=0.378 CR=0.192 CF1=0.255|
|L,U,T:1600,,1191（标签数：103）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=44.848 OP=0.018 OR=0.661 OF1=0.722 CP=0.475 CR=0.316 CF1=0.380|
|L,U,T:6400,,1191（标签数：103）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=53.451 OP=0.018 OR=0.711 OF1=0.760 CP=0.559 CR=0.390 CF1=0.460|
|---|---|---|---|
|L,U,T: 200,800,1191（标签数：103）|Bert微调+多注意力GAN3|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T: 400,800,1191（标签数：103）|Bert微调+多注意力GAN3|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:1600,800,1191（标签数：103）|Bert微调+多注意力GAN3|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:6400,800,1191（标签数：103）|Bert微调+多注意力GAN3|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|
- GAN3效果补好

## Stack Overflow数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,（标签数：100）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=38.436 OP=0.314 OR=0.290 OF1=0.420 CP=0.477 CR=0.268 CF1=0.343|
|L,U,T:200,400,（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=38.793 OP=0.295 OR=0.200 OF1=0.314 CP=0.370 CR=0.141 CF1=0.205|
|L,U,T:400,,（标签数：100）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:400,400,（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=53.956 OP=0.429 OR=0.368 OF1=0.491 CP=0.606 CR=0.354 CF1=0.447|
|L,U,T:1600,,（标签数：100）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=63.090 OP=0.489 OR=0.579 OF1=0.644 CP=0.701 CR=0.578 CF1=0.634|
|L,U,T:1600,400,（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=63.564 OP=0.483 OR=0.580 OF1=0.637 CP=0.728 CR=0.589 CF1=0.651|
|L,U,T:6400,,（标签数：100）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:6400,400,（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

# 实验结果 0113-
## Stack Overflow数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 200,,537（标签数：50）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=47.684 OP=0.434 OR=0.362 OF1=0.502 CP=0.617 CR=0.355 CF1=0.451|
|L,U,T: 400,,537（标签数：50）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=55.497 OP=0.518 OR=0.452 OF1=0.577 CP=0.690 CR=0.438 CF1=0.536|
|L,U,T:1600,,537（标签数：50）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=65.899 OP=0.559 OR=0.626 OF1=0.671 CP=0.720 CR=0.621 CF1=0.667|
|L,U,T:6400,,537（标签数：50）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=70.768 OP=0.594 OR=0.657 OF1=0.689 CP=0.780 CR=0.637 CF1=0.701|
|L,U,T:12800,,537（标签数：50）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=73.578 OP=0.607 OR=0.653 OF1=0.697 CP=0.744 CR=0.651 CF1=0.695|
|L,U,T:全部,,537（标签数：50）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=77.419 OP=0.619 OR=0.648 OF1=0.731 CP=0.858 CR=0.637 CF1=0.731|
|---|---|---|---|
|L,U,T: 200,1600,537（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=50.757 OP=0.477 OR=0.397 OF1=0.529 CP=0.627 CR=0.386 CF1=0.478|
|L,U,T: 400,1600,537（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=56.251 OP=0.529 OR=0.459 OF1=0.580 CP=0.720 CR=0.451 CF1=0.555|
|L,U,T:1600,1600,537（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=66.488 OP=0.564 OR=0.583 OF1=0.658 CP=0.757 CR=0.587 CF1=0.661|
|L,U,T:6400,1600,537（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=70.845 OP=0.581 OR=0.645 OF1=0.686 CP=0.724 CR=0.666 CF1=0.694|
|L,U,T:12800,1600,537（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=73.928 OP=0.588 OR=0.645 OF1=0.697 CP=0.778 CR=0.656 CF1=0.712|
|L,U,T:全部,1600,537（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|---|---|---|---|

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 200,,760（标签数：100）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=41.596 OP=0.338 OR=0.282 OF1=0.408 CP=0.466 CR=0.243 CF1=0.319|
|L,U,T: 400,,760（标签数：100）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=53.799 OP=0.414 OR=0.410 OF1=0.535 CP=0.621 CR=0.400 CF1=0.486|
|L,U,T:1600,,760（标签数：100）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=63.976 OP=0.486 OR=0.560 OF1=0.637 CP=0.705 CR=0.582 CF1=0.637|
|L,U,T:6400,,760（标签数：100）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=68.603 OP=0.529 OR=0.633 OF1=0.683 CP=0.737 CR=0.637 CF1=0.684|
|L,U,T:12800,,537（标签数：100）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=69.506 OP=0.534 OR=0.609 OF1=0.683 CP=0.763 CR=0.620 CF1=0.684|
|L,U,T:,,537（标签数：100）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|---|---|---|---|
|L,U,T: 200,1600,760（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=45.046 OP=0.376 OR=0.280 OF1=0.415 CP=0.464 CR=0.264 CF1=0.337|
|L,U,T: 400,1600,760（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=54.898 OP=0.443 OR=0.436 OF1=0.544 CP=0.604 CR=0.422 CF1=0.497|
|L,U,T:1600,1600,760（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=63.460 OP=0.497 OR=0.574 OF1=0.639 CP=0.705 CR=0.604 CF1=0.651|
|L,U,T:6400,1600,760（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=68.527 OP=0.511 OR=0.630 OF1=0.676 CP=0.740 CR=0.635 CF1=0.684|
|L,U,T:12800,1600,537（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=69.343 OP=0.538 OR=0.611 OF1=0.687 CP=0.794 CR=0.602 CF1=0.685|
|L,U,T:,1600,537（标签数：100）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||

|---|---|---|---|


|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 200,,877（标签数：150）|Bert微调+多注意力|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=34.580 OP=0.233 OR=0.180 OF1=0.284 CP=0.296 CR=0.142 CF1=0.192|
|L,U,T: 400,,877（标签数：150）|Bert微调+多注意力|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=45.021 OP=0.333 OR=0.339 OF1=0.460 CP=0.504 CR=0.306 CF1=0.381|
|L,U,T:1600,,877（标签数：150）|Bert微调+多注意力|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=57.114 OP=0.422 OR=0.555 OF1=0.597 CP=0.626 CR=0.533 CF1=0.575|
|L,U,T:6400,,877（标签数：150）|Bert微调+多注意力|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=63.520 OP=0.462 OR=0.591 OF1=0.636 CP=0.652 CR=0.579 CF1=0.613|
|L,U,T:12800,,877（标签数：150）|Bert微调+多注意力|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=65.019 OP=0.477 OR=0.625 OF1=0.665 CP=0.684 CR=0.600 CF1=0.639|
|---|---|---|---|
|L,U,T: 200,1600,877（标签数：150）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=41.755 OP=0.301 OR=0.211 OF1=0.326 CP=0.288 CR=0.166 CF1=0.211|
|L,U,T: 400,1600,877（标签数：150）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=49.022 OP=0.350 OR=0.376 OF1=0.480 CP=0.461 CR=0.334 CF1=0.387|
|L,U,T:1600,1600,877（标签数：150）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=56.177 OP=0.423 OR=0.565 OF1=0.597 CP=0.603 CR=0.546 CF1=0.573|
|L,U,T:6400,1600,877（标签数：150）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=63.290 OP=0.469 OR=0.608 OF1=0.644 CP=0.655 CR=0.590 CF1=0.621|
|L,U,T:12800,1600,877（标签数：150）|Bert微调+多注意力+GAN|epoch:48;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=65.010 OP=0.464 OR=0.619 OF1=0.661 CP=0.678 CR=0.622 CF1=0.649|
|---|---|---|---|

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 200,,972（标签数：200）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=32.445 OP=0.195 OR=0.140 OF1=0.231 CP=0.291 CR=0.118 CF1=0.168|
|L,U,T: 400,,972（标签数：200）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=46.142 OP=0.294 OR=0.297 OF1=0.420 CP=0.401 CR=0.280 CF1=0.330|
|L,U,T:1600,,972（标签数：200）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=57.902 OP=0.401 OR=0.508 OF1=0.585 CP=0.614 CR=0.524 CF1=0.565|
|L,U,T:6400,,972（标签数：200）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=61.983 OP=0.432 OR=0.564 OF1=0.628 CP=0.651 CR=0.562 CF1=0.603|
|L,U,T:12800,,537（标签数：200）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=63.727 OP=0.443 OR=0.555 OF1=0.625 CP=0.643 CR=0.559 CF1=0.598|
|L,U,T:,,537（标签数：200）|Bert微调+多注意力|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|---|---|---|---|
|L,U,T: 200,1600,972（标签数：200）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=37.993 OP=0.267 OR=0.164 OF1=0.263 CP=0.207 CR=0.135 CF1=0.163|
|L,U,T: 400,1600,972（标签数：200）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=46.461 OP=0.313 OR=0.297 OF1=0.411 CP=0.389 CR=0.286 CF1=0.330|
|L,U,T:1600,1600,972（标签数：200）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=53.887 OP=0.383 OR=0.489 OF1=0.560 CP=0.582 CR=0.481 CF1=0.527|
|L,U,T:6400,1600,972（标签数：200）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=60.887 OP=0.422 OR=0.562 OF1=0.623 CP=0.628 CR=0.578 CF1=0.602|
|L,U,T:12800,1600,537（标签数：200）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=64.303 OP=0.447 OR=0.581 OF1=0.633 CP=0.643 CR=0.602 CF1=0.622|
|L,U,T:,1600,537（标签数：200）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|---|---|---|---|


|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 400,,1065（标签数：300）|Bert微调+多注意力|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=43.337 OP=0.254 OR=0.261 OF1=0.377 CP=0.342 CR=0.200 CF1=0.253|
|L,U,T:1600,,1065（标签数：300）|Bert微调+多注意力|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=54.558 OP=0.353 OR=0.527 OF1=0.560 CP=0.532 CR=0.495 CF1=0.513|
|L,U,T:6400,,1065（标签数：300）|Bert微调+多注意力|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=58.065 OP=0.362 OR=0.490 OF1=0.576 CP=0.512 CR=0.509 CF1=0.510|
|L,U,T:12800,,1065（标签数：300）|Bert微调+多注意力|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=59.426 OP=0.367 OR=0.492 OF1=0.597 CP=0.545 CR=0.476 CF1=0.508|
|---|---|---|---|
|L,U,T: 400,1600,1065（标签数：300）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=38.000 OP=0.231 OR=0.214 OF1=0.320 CP=0.237 CR=0.173 CF1=0.200|
|L,U,T:1600,1600,1065（标签数：300）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=48.587 OP=0.295 OR=0.402 OF1=0.493 CP=0.430 CR=0.364 CF1=0.394|
|L,U,T:6400,1600,1065（标签数：300）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:12800,1600,1065（标签数：300）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=58.706 OP=0.368 OR=0.512 OF1=0.599 CP=0.603 CR=0.503 CF1=0.548|
|---|---|---|---|


|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: ,,（标签数：全部）|Bert微调+多注意力|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|---|---|---|---|
|L,U,T:,,（标签数：全部）|Bert微调+多注意力+GAN|epoch:46;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=54.336 OP=0.353 OR=0.523 OF1=0.599 CP=0.520 CR=0.447 CF1=0.481|


- 另外，进行了采用sigmoid进行生成对抗，效果不太好

## freecode数据集
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 200,,8091（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=25.208 OP=0.066 OR=0.268 OF1=0.372 CP=0.316 CR=0.155 CF1=0.208|
|L,U,T: 400,,8091（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=32.931 OP=0.094 OR=0.316 OF1=0.434 CP=0.570 CR=0.189 CF1=0.284|
|L,U,T:1600,,8091（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=40.906 OP=0.111 OR=0.413 OF1=0.516 CP=0.588 CR=0.306 CF1=0.403|
|L,U,T:6400,,8091（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=46.065 OP=0.115 OR=0.508 OF1=0.565 CP=0.567 CR=0.401 CF1=0.469|
|L,U,T:全部,,8091（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=53.168 OP=0.128 OR=0.455 OF1=0.557 CP=0.678 CR=0.369 CF1=0.478|
|---|---|---|---|
|L,U,T: 200,1600,8091（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=29.913 OP=0.082 OR=0.362 OF1=0.443 CP=0.458 CR=0.185 CF1=0.264|
|L,U,T: 400,1600,8091（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=34.433 OP=0.103 OR=0.376 OF1=0.474 CP=0.517 CR=0.236 CF1=0.324|
|L,U,T:1600,1600,8091（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=42.049 OP=0.111 OR=0.415 OF1=0.515 CP=0.575 CR=0.326 CF1=0.416|
|L,U,T:6400,1600,8091（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=48.542 OP=0.122 OR=0.518 OF1=0.571 CP=0.532 CR=0.438 CF1=0.481|
|L,U,T:全部,1600,8091（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=53.771 OP=0.123 OR=0.565 OF1=0.571 CP=0.529 CR=0.485 CF1=0.506|
|---|---|---|---|

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 200,,8482（标签数：100）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=18.097 OP=0.037 OR=0.283 OF1=0.373 CP=0.270 CR=0.123 CF1=0.169|
|L,U,T: 400,,8482（标签数：100）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=22.379 OP=0.040 OR=0.307 OF1=0.398 CP=0.372 CR=0.142 CF1=0.205|
|L,U,T:1600,,8482（标签数：100）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=32.866 OP=0.056 OR=0.401 OF1=0.483 CP=0.481 CR=0.261 CF1=0.338|
|L,U,T:6400,,8482（标签数：100）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=38.877 OP=0.064 OR=0.438 OF1=0.511 CP=0.513 CR=0.309 CF1=0.386|
|L,U,T:全部,,8482（标签数：100）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=44.660 OP=0.069 OR=0.407 OF1=0.518 CP=0.484 CR=0.345 CF1=0.403|
|---|---|---|---|
|L,U,T: 200,1600,8482（标签数：100）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=20.308 OP=0.044 OR=0.226 OF1=0.331 CP=0.243 CR=0.063 CF1=0.100|
|L,U,T: 400,1600,8482（标签数：100）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=24.881 OP=0.049 OR=0.263 OF1=0.369 CP=0.364 CR=0.109 CF1=0.167|
|L,U,T:1600,1600,8482（标签数：100）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=34.402 OP=0.056 OR=0.433 OF1=0.488 CP=0.510 CR=0.251 CF1=0.336|
|L,U,T:6400,1600,8482（标签数：100）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=40.266 OP=0.065 OR=0.476 OF1=0.524 CP=0.431 CR=0.396 CF1=0.413|
|L,U,T:全部,1600,8482（标签数：100）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=46.003 OP=0.071 OR=0.480 OF1=0.559 CP=0.543 CR=0.378 CF1=0.445|
|---|---|---|---|

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 200,,（标签数：150）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T: 400,,（标签数：150）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:1600,,（标签数：150）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:6400,,（标签数：150）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:12800,,（标签数：150）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:全部,,（标签数：150）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|---|---|---|---|
|L,U,T: 200,1600,（标签数：150）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T: 400,1600,（标签数：150）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:1600,1600,（标签数：150）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:6400,1600,（标签数：150）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:12800,1600,（标签数：150）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:全部,1600,（标签数：150）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|---|---|---|---|

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 200,,8906（标签数：200）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=7.584 OP=0.017 OR=0.194 OF1=0.277 CP=0.151 CR=0.032 CF1=0.053|
|L,U,T: 400,,8906（标签数：200）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=10.989 OP=0.024 OR=0.244 OF1=0.320 CP=0.205 CR=0.057 CF1=0.089|
|L,U,T:1600,,8906（标签数：200）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=24.863 OP=0.032 OR=0.342 OF1=0.433 CP=0.379 CR=0.144 CF1=0.209|
|L,U,T:6400,,8906（标签数：200）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=31.628 OP=0.039 OR=0.380 OF1=0.454 CP=0.406 CR=0.243 CF1=0.304|
|L,U,T:全部,,8906（标签数：200）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=34.962 OP=0.036 OR=0.394 OF1=0.468 CP=0.403 CR=0.247 CF1=0.307|
|---|---|---|---|
|L,U,T: 200,1600,8906（标签数：200）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=8.519 OP=0.021 OR=0.146 OF1=0.233 CP=0.040 CR=0.014 CF1=0.020|
|L,U,T: 400,1600,8906（标签数：200）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=14.030 OP=0.028 OR=0.190 OF1=0.293 CP=0.100 CR=0.029 CF1=0.045|
|L,U,T:1600,1600,8906（标签数：200）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=24.970 OP=0.033 OR=0.298 OF1=0.406 CP=0.319 CR=0.108 CF1=0.161|
|L,U,T:6400,1600,8906（标签数：200）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=30.676 OP=0.038 OR=0.449 OF1=0.485 CP=0.390 CR=0.274 CF1=0.322|
|L,U,T:全部,1600,8906（标签数：200）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=37.382 OP=0.043 OR=0.409 OF1=0.507 CP=0.511 CR=0.242 CF1=0.328|
|---|---|---|---|

#0119-

##Stack Overflow
数据集是过滤掉tag_freq<200的，剩254个tag

小数量tag优先划分的数据集（可能是代码写错了，实际是从大开始的）
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,（标签数：50）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=36.128 OP=0.242 OR=0.355 OF1=0.493 CP=0.561 CR=0.262 CF1=0.357|
|L,U,T:200,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=39.578 OP=0.245 OR=0.338 OF1=0.468 CP=0.597 CR=0.239 CF1=0.341|
|L,U,T:400,,（标签数：50）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=43.350 OP=0.264 OR=0.415 OF1=0.546 CP=0.638 CR=0.334 CF1=0.438|
|L,U,T:400,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=43.530 OP=0.265 OR=0.413 OF1=0.535 CP=0.621 CR=0.334 CF1=0.435|
|L,U,T:1600,,（标签数：50）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=57.329 OP=0.302 OR=0.519 OF1=0.622 CP=0.736 CR=0.491 CF1=0.589|
|L,U,T:1600,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=56.917 OP=0.300 OR=0.543 OF1=0.619 CP=0.685 CR=0.523 CF1=0.593|
|L,U,T:6400,,（标签数：50）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:6400,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

大数量tag优先划分的数据集（可能是代码写错了，实际是从小开始的）
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,（标签数：50）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=36.505 OP=0.222 OR=0.338 OF1=0.466 CP=0.592 CR=0.251 CF1=0.352|
|L,U,T:200,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=41.553 OP=0.232 OR=0.361 OF1=0.480 CP=0.563 CR=0.285 CF1=0.378|
|L,U,T:400,,（标签数：50）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=45.956 OP=0.264 OR=0.432 OF1=0.552 CP=0.684 CR=0.371 CF1=0.481|
|L,U,T:400,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=48.006 OP=0.264 OR=0.433 OF1=0.540 CP=0.673 CR=0.383 CF1=0.488|
|L,U,T:1600,,（标签数：50）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|56|
|L,U,T:1600,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:6400,,（标签数：50）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:6400,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

#0120-

##Stack Overflow
数据集是过滤掉tag_freq<200的，剩254个tag
小数量tag优先划分的数据集
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=39.699,40.456 OP=0.296,0.289 OR=0.335,0.350 OF1=0.457,0.468 CP=0.643,0.571 CR=0.272,0.309 CF1=0.383,0.401|
|L,U,T:200,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=45.453,45.049 OP=0.348,0.334  OR=0.367,0.354 OF1=0.494,0.481 CP=0.655,0.612 CR=0.330,0.311 CF1=0.439,0.413|
|L,U,T:400,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=50.727,51.111 OP=0.365,0.366 OR=0.466,0.455 OF1=0.568,0.569 CP=0.706,0.742 CR=0.438,0.419 CF1=0.540,0.535|
|L,U,T:400,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=53.796,54.958 OP=0.390,0.397 OR=0.452,0.471 OF1=0.567,0.581 CP=0.744,0.745 CR=0.431,0.452 CF1=0.546,0.562|
|L,U,T:1600,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=64.562,64.880 OP=0.447,0.448 OR=0.569,0.579 OF1=0.659,0.663 CP=0.778,0.770 CR=0.556,0.567 CF1=0.649,0.653|
|L,U,T:1600,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:30;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=64.179,64.086 OP=0.437,0.435 OR=0.554,0.554 OF1=0.638,0.637 CP=0.757,0.751 CR=0.547,0.552 CF1=0.635,0.636|
|L,U,T:6400,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:6400,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

大数量tag优先划分的数据集
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=39.491,38.591 OP=0.308,0.293 OR=0.344,0.324 OF1=0.466,0.445 CP=0.624,0.631 CR=0.279,0.284 CF1=0.38,0.391|
|L,U,T:200,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=43.181,45.540 OP=0.335,0.345 OR=0.298,0.359 OF1=0.435,0.489 CP=0.635,0.683 CR=0.247,0.315 CF1=0.355,0.431|
|L,U,T:400,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=44.987 OP=0.343 OR=0.372 OF1=0.505 CP=0.704 CR=0.306 CF1=0.427|
|L,U,T:400,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=53.926 OP=0.402 OR=0.475 OF1=0.588 CP=0.734 CR=0.450 CF1=0.558|
|L,U,T:1600,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=64.468 OP=0.454 OR=0.584 OF1=0.671 CP=0.777 CR=0.576 CF1=0.662|
|L,U,T:1600,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:6400,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:6400,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|


其他试验：
- 有一次（不去频率最大的标签）从小和从大分割出的数据集gan性能都不好，并且200时只有36。但有一次这样的数据又是正常的。

全部数据集
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:97454,,25508（标签数：254）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=51.437 OP=0.146 OR=0.493 OF1=0.603 CP=0.610 CR=0.438 CF1=0.510|
|---|---|---|---|
|L,U,T:97454,1600,25508（标签数：254）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=54.065 OP=0.151 OR=0.541 OF1=0.629 CP=0.639 CR=0.468 CF1=0.540|
|---|---|---|---|

##AAPD
数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:全部,,（标签数：）|Bert微调+多注意力|epoch:45;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=56.267 OP=0.000 OR=0.644 OF1=0.681 CP=0.574 CR=0.473 CF1=0.519|
|---|---|---|---|
|L,U,T:全部,1600,（标签数：）|Bert微调+多注意力+GAN|epoch:20;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=59.075 OP=0.005 OR=0.622 OF1=0.693 CP=0.635 CR=0.481 CF1=0.547|
|---|---|---|---|

##Freecode
数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:32890,,7813（标签数：）|Bert微调+多注意力|epoch:50;epoch_step:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=43.497 OP=0.050 OR=0.405 OF1=0.515 CP=0.444 CR=0.341 CF1=0.386|
|---|---|---|---|
|L,U,T:32890,1600,7813（标签数：）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=44.465 OP=0.051 OR=0.458 OF1=0.540 CP=0.557 CR=0.343 CF1=0.424|
|---|---|---|---|

##对Stack Overflow的数据集选取

频率大于200后，全部tag
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=36.361 OP=0.233 OR=0.354 OF1=0.490 CP=0.526 CR=0.263 CF1=0.350|
|L,U,T:200,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=37.376 OP=0.230 OR=0.354 OF1=0.481 CP=0.516 CR=0.245 CF1=0.332|
|L,U,T:400,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=44.851 OP=0.256 OR=0.439 OF1=0.556 CP=0.631 CR=0.362 CF1=0.460|
|L,U,T:400,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=47.218 OP=0.263 OR=0.430 OF1=0.551 CP=0.662 CR=0.364 CF1=0.470|
|L,U,T:1600,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:1600,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:30;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

频率大于200后，去除频率最大的两个tag
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|42，0.5，0.42|
|L,U,T:200,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=44.873 OP=0.275 OR=0.367 OF1=0.497 CP=0.627 CR=0.283 CF1=0.390|
|L,U,T:400,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=48.642 OP=0.304 OR=0.445 OF1=0.576 CP=0.714 CR=0.384 CF1=0.500|
|L,U,T:400,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=50.098 OP=0.303 OR=0.441 OF1=0.566 CP=0.702 CR=0.389 CF1=0.500|
|L,U,T:1600,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=59.645 OP=0.341 OR=0.548 OF1=0.646 CP=0.711 CR=0.538 CF1=0.612|
|L,U,T:1600,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:30;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=58.790 OP=0.340 OR=0.568 OF1=0.638 CP=0.716 CR=0.527 CF1=0.607|
|---|---|---|---|

频率大于200后，去除频率大于5000的tag
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=41.924 OP=0.295 OR=0.344 OF1=0.473 CP=0.657 CR=0.296 CF1=0.408|
|L,U,T:200,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=47.307 OP=0.345 OR=0.337 OF1=0.475 CP=0.698 CR=0.302 CF1=0.422|
|L,U,T:400,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=50.533 OP=0.360 OR=0.425 OF1=0.555 CP=0.741 CR=0.394 CF1=0.515|
|L,U,T:400,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=53.322 OP=0.373 OR=0.436 OF1=0.561 CP=0.748 CR=0.412 CF1=0.532|
|L,U,T:1600,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01||
|L,U,T:1600,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:30;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

频率大于200后，去除频率大于2000的tag
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:200,,5198（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=41.274 OP=0.347 OR=0.265 OF1=0.401 CP=0.684 CR=0.249 CF1=0.365|
|L,U,T:200,1600,5198（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=47.039 OP=0.391 OR=0.295 OF1=0.431 CP=0.702 CR=0.267 CF1=0.387|
|L,U,T:400,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=54.740 OP=0.438 OR=0.453 OF1=0.576 CP=0.769 CR=0.441 CF1=0.560|
|L,U,T:400,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=57.271 OP=0.461 OR=0.495 OF1=0.602 CP=0.752 CR=0.478 CF1=0.584|
|L,U,T:1600,,（标签数：50）|Bert微调+多注意力|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=67.767 OP=0.520 OR=0.595 OF1=0.683 CP=0.793 CR=0.594 CF1=0.679|
|L,U,T:1600,1600,（标签数：50）|Bert微调+多注意力+GAN|epoch:70;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=67.311 OP=0.512 OR=0.590 OF1=0.676 CP=0.784 CR=0.588 CF1=0.672|
|---|---|---|---|

其他试验：
- 过滤5000以上的，200：map=40.874 OP=0.300 OR=0.335 OF1=0.467 CP=0.643 CR=0.308 CF1=0.417
- 过滤2000以上的 200： map=45.123 OP=0.377 OR=0.342 OF1=0.479 CP=0.655 CR=0.339 CF1=0.447；Gan map=46.214 OP=0.389 OR=0.270 OF1=0.405 CP=0.692 CR=0.245 CF1=0.362
-不过滤最大标签 全部数据训练 2 epoch：map=51.778 OP=0.143 OR=0.501 OF1=0.606 CP=0.590 CR=0.427 CF1=0.496
-过滤前两个 全部数据训练 2 epoch：map=53.747 OP=0.173 OR=0.469 OF1=0.599 CP=0.588 CR=0.385 CF1=0.465

#0121-
##Stack Overflow按最终确认的数据进行的试验（去除频率小于200和大于2000的tag）

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 200,    ,5198（标签数：50）|Bert微调+多注意力    |epoch: 50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=45.644 OP=0.359 OR=0.353 OF1=0.474 CP=0.692 CR=0.331 CF1=0.448|
|L,U,T: 200,1600,5198（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=49.122 OP=0.407 OR=0.351 OF1=0.487 CP=0.706 CR=0.329 CF1=0.449|
|L,U,T: 400,    ,5198（标签数：50）|Bert微调+多注意力    |epoch: 50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=54.804 OP=0.439 OR=0.428 OF1=0.560 CP=0.757 CR=0.420 CF1=0.540|
|L,U,T: 400,1600,5198（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=58.001 OP=0.459 OR=0.498 OF1=0.605 CP=0.744 CR=0.483 CF1=0.586|
|L,U,T:1600,    ,5198（标签数：50）|Bert微调+多注意力    |epoch: 50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01||
|L,U,T:1600,1600,5198（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=66.985 OP=0.508 OR=0.583 OF1=0.672 CP=0.787 CR=0.581 CF1=0.669|
|L,U,T:6400,    ,5198（标签数：50）|Bert微调+多注意力    |epoch:  8;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=65.639 OP=0.473 OR=0.497 OF1=0.619 CP=0.823 CR=0.495 CF1=0.618|
|L,U,T:6400,1600,5198（标签数：50）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 200,    ,7873（标签数：200）|Bert微调+多注意力    |epoch: 50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=23.148 OP=0.128 OR=0.103 OF1=0.179 CP=0.338 CR=0.081 CF1=0.130|
|L,U,T: 200,1600,7873（标签数：200）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=29.326 OP=0.192 OR=0.131 OF1=0.222 CP=0.230 CR=0.111 CF1=0.150|
|L,U,T: 400,    ,7873（标签数：200）|Bert微调+多注意力    |epoch: 50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=37.064 OP=0.211 OR=0.281 OF1=0.402 CP=0.508 CR=0.230 CF1=0.317|
|L,U,T: 400,1600,7873（标签数：200）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=40.683 OP=0.242 OR=0.331 OF1=0.445 CP=0.500 CR=0.289 CF1=0.366|
|L,U,T:1600,    ,7873（标签数：200）|Bert微调+多注意力    |epoch: 50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01||
|L,U,T:1600,1600,7873（标签数：200）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:6400,    ,7873（标签数：200）|Bert微调+多注意力    |epoch: 50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01||
|L,U,T:6400,1600,7873（标签数：200）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

#0122-

##Stack Overflow 1000
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 500,    ,6416（标签数：207）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=40.745 OP=0.261 OR=0.289 OF1=0.410 CP=0.535 CR=0.259 CF1=0.349|
|L,U,T: 500,1600,6416（标签数：207）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=43.747 OP=0.275 OR=0.322 OF1=0.439 CP=0.484 CR=0.295 CF1=0.366|
|L,U,T:1000,    ,6416（标签数：207）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=52.216 OP=0.341 OR=0.432 OF1=0.547 CP=0.652 CR=0.414 CF1=0.506|
|L,U,T:1000,1600,6416（标签数：207）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=48.016 OP=0.317 OR=0.442 OF1=0.535 CP=0.610 CR=0.423 CF1=0.499|
|L,U,T:2000,    ,6416（标签数：207）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=56.046 OP=0.364 OR=0.552 OF1=0.617 CP=0.678 CR=0.541 CF1=0.602|
|L,U,T:2000,1600,6416（标签数：207）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=52.140 OP=0.344 OR=0.532 OF1=0.594 CP=0.639 CR=0.520 CF1=0.573|
|L,U,T:5000,    ,6416（标签数：207）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01||
|L,U,T:5000,1600,6416（标签数：207）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

##Stack Overflow 5000
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 500,    ,10004（标签数：243）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=35.529 OP=0.172 OR=0.332 OF1=0.447 CP=0.469 CR=0.248 CF1=0.324|
|L,U,T: 500,1600,10004（标签数：243）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=36.811 OP=0.182 OR=0.301 OF1=0.423 CP=0.406 CR=0.223 CF1=0.288|
|L,U,T:1000,    ,10004（标签数：243）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=45.520 OP=0.215 OR=0.432 OF1=0.538 CP=0.565 CR=0.351 CF1=0.433|
|L,U,T:1000,1600,10004（标签数：243）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=41.770 OP=0.208 OR=0.380 OF1=0.499 CP=0.506 CR=0.304 CF1=0.380|
|L,U,T:2000,    ,10004（标签数：243）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=50.305 OP=0.232 OR=0.501 OF1=0.589 CP=0.609 CR=0.484 CF1=0.539|
|L,U,T:2000,1600,10004（标签数：243）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=46.799 OP=0.223 OR=0.466 OF1=0.564 CP=0.577 CR=0.410 CF1=0.479|
|L,U,T:5000,    ,10004（标签数：243）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01||
|L,U,T:5000,1600,10004（标签数：243）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

##Stack Overflow > 300
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 500,    ,12553（标签数：180）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=32.296 OP=0.138 OR=0.382 OF1=0.498 CP=0.486 CR=0.217 CF1=0.300|
|L,U,T: 500,1600,12553（标签数：180）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=37.987 OP=0.142 OR=0.382 OF1=0.494 CP=0.412 CR=0.227 CF1=0.293|
|L,U,T:1000,    ,12553（标签数：180）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=42.610 OP=0.153 OR=0.441 OF1=0.552 CP=0.545 CR=0.340 CF1=0.419|
|L,U,T:1000,1600,12553（标签数：180）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=42.220 OP=0.152 OR=0.436 OF1=0.545 CP=0.548 CR=0.326 CF1=0.409|
|L,U,T:2000,    ,12553（标签数：180）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01||
|L,U,T:2000,1600,12553（标签数：180）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:5000,    ,12553（标签数：180）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01||
|L,U,T:5000,1600,12553（标签数：180）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

##Stack Overflow > 400
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 500,    ,12303（标签数：140）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=36.410 OP=0.163 OR=0.389 OF1=0.515 CP=0.481 CR=0.279 CF1=0.353|
|L,U,T: 500,1600,12303（标签数：140）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=39.769 OP=0.165 OR=0.400 OF1=0.518 CP=0.519 CR=0.281 CF1=0.364|
|L,U,T:1000,    ,12303（标签数：140）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=45.182 OP=0.172 OR=0.453 OF1=0.561 CP=0.589 CR=0.374 CF1=0.457|
|L,U,T:1000,1600,12303（标签数：140）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:2000,    ,12303（标签数：140）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01||
|L,U,T:2000,1600,12303（标签数：140）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:5000,    ,12303（标签数：140）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01||
|L,U,T:5000,1600,12303（标签数：140）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|---|---|---|---|

#0123-
##Stack Overflow 2000

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 500,    ,8019（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01||
|L,U,T: 500,1600,8019（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=39.283 OP=0.216 OR=0.263 OF1=0.382 CP=0.429 CR=0.237 CF1=0.306|
|L,U,T:1000,    ,8019（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=48.834 OP=0.268 OR=0.388 OF1=0.513 CP=0.601 CR=0.363 CF1=0.453|
|L,U,T:1000,1600,8019（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001||
|L,U,T:2000,    ,8019（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=53.328 OP=0.292 OR=0.506 OF1=0.588 CP=0.658 CR=0.488 CF1=0.560|
|L,U,T:2000,1600,8019（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=49.546 OP=0.273 OR=0.479 OF1=0.567 CP=0.615 CR=0.458 CF1=0.525|
|L,U,T:5000,    ,8019（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=56.753 OP=0.303 OR=0.526 OF1=0.612 CP=0.643 CR=0.535 CF1=0.584|
|L,U,T:5000,1600,8019（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=55.247 OP=0.297 OR=0.507 OF1=0.598 CP=0.672 CR=0.494 CF1=0.569|
|---|---|---|---|

其他试验，对stackoverflow的tag按频率切分为前114和后114个，结果：
- 前114个：map=27.115 OP=0.191 OR=0.167 OF1=0.270 CP=0.436 CR=0.131 CF1=0.201
- 后114个：map=41.094 OP=0.360 OR=0.238 OF1=0.363 CP=0.494 CR=0.227 CF1=0.311
这种现象在programmerweb数据集也存在，原因估计是前114个tag样本虽然多，但tag之间更不平衡。
- 另外，又做了频率前40的tag的试验：map=48.881 OP=0.396 OR=0.375 OF1=0.515 CP=0.712 CR=0.372 CF1=0.489
 
数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:全部71285,,8019（标签数：）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=59.499 OP=0.306 OR=0.526 OF1=0.620 CP=0.683 CR=0.509 CF1=0.584|
|---|---|---|---|
|L,U,T:全部71285,1600,8019（标签数：）|Bert微调+多注意力+GAN|epoch:20;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=61.509 OP=0.313 OR=0.591 OF1=0.645 CP=0.682 CR=0.575 CF1=0.624|
|---|---|---|---|

MLPBert：完整
- （60epoch 50step）map=47.215 OP=0.277 OR=0.333 OF1=0.475 CP=0.616 CR=0.298 CF1=0.402


600-2000:
MLPBert: 60,50 map=72.459 OP=0.509 OR=0.596 OF1=0.684 CP=0.786 CR=0.610 CF1=0.687

##AAPD

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 500,    ,6207（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=32.098 OP=0.001 OR=0.444 OF1=0.550 CP=0.356 CR=0.280 CF1=0.314|
|L,U,T: 500,1600,6207（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=35.993 OP=0.000 OR=0.462 OF1=0.559 CP=0.462 CR=0.248 CF1=0.323|
|L,U,T:1000,    ,6207（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=38.756 OP=0.001 OR=0.523 OF1=0.601 CP=0.493 CR=0.303 CF1=0.375|
|L,U,T:1000,1600,6207（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=39.911 OP=0.002 OR=0.508 OF1=0.598 CP=0.531 CR=0.301 CF1=0.384|
|L,U,T:2000,    ,6207（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=42.142 OP=0.001 OR=0.553 OF1=0.625 CP=0.495 CR=0.354 CF1=0.413|
|L,U,T:2000,1600,6207（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=45.619 OP=0.020 OR=0.555 OF1=0.626 CP=0.524 CR=0.376 CF1=0.438|
|L,U,T:5000,    ,6207（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=47.499 OP=0.004 OR=0.583 OF1=0.648 CP=0.568 CR=0.399 CF1=0.469|
|L,U,T:5000,1600,6207（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=50.101 OP=0.036 OR=0.588 OF1=0.649 CP=0.554 CR=0.420 CF1=0.478|
|---|---|---|---|

数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:全部,,（标签数：）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=54.725 OP=0.000 OR=0.609 OF1=0.685 CP=0.551 CR=0.468 CF1=0.506|
|---|---|---|---|
|L,U,T:全部,1600,（标签数：）|Bert微调+多注意力+GAN|epoch:118;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=58.848 OP=0.008 OR=0.654 OF1=0.702 CP=0.637 CR=0.480 CF1=0.547|
|---|---|---|---|

##Freecode

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 500,    ,3905（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=23.854 OP=0.030 OR=0.351 OF1=0.442 CP=0.354 CR=0.163 CF1=0.223|
|L,U,T: 500,1600,3905（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=26.594 OP=0.037 OR=0.302 OF1=0.412 CP=0.294 CR=0.115 CF1=0.165|
|L,U,T:1000,    ,3905（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=29.330 OP=0.033 OR=0.360 OF1=0.451 CP=0.440 CR=0.192 CF1=0.267|
|L,U,T:1000,1600,3905（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=32.020 OP=0.038 OR=0.410 OF1=0.479 CP=0.428 CR=0.208 CF1=0.280|
|L,U,T:2000,    ,3905（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=33.512 OP=0.037 OR=0.402 OF1=0.485 CP=0.482 CR=0.247 CF1=0.327|
|L,U,T:2000,1600,3905（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=36.126 OP=0.038 OR=0.411 OF1=0.490 CP=0.511 CR=0.250 CF1=0.336|
|L,U,T:5000,    ,3905（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|map=37.835 OP=0.039 OR=0.430 OF1=0.497 CP=0.505 CR=0.301 CF1=0.377|
|L,U,T:5000,1600,3905（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=39.676 OP=0.039 OR=0.445 OF1=0.517 CP=0.501 CR=0.305 CF1=0.379|
|---|---|---|---|

数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T:36798,,3905（标签数：）|Bert微调+多注意力|epoch:60;epoch_step:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|map=40.172 OP=0.039 OR=0.418 OF1=0.495 CP=0.498 CR=0.309 CF1=0.381|
|---|---|---|---|
|L,U,T:36798,1600,3905（标签数：）|Bert微调+多注意力+GAN|epoch:94;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|map=46.264 OP=0.043 OR=0.374 OF1=0.499 CP=0.629 CR=0.252 CF1=0.360|
|---|---|---|---|

其他试验，对StackOverflow，AAPD，Freecode进行了小样本训练：
- StackOverflow 200 80epoch
map=17.767 OP=0.071 OR=0.051 OF1=0.093 CP=0.296 CR=0.061 CF1=0.102
map=20.930 OP=0.118 OR=0.037 OF1=0.070 CP=0.118 CR=0.035 CF1=0.054
- AAPD 200 48epoch
map=24.772 OP=0.002 OR=0.362 OF1=0.474 CP=0.331 CR=0.152 CF1=0.208
map=28.122 OP=0.000 OR=0.402 OF1=0.499 CP=0.394 CR=0.166 CF1=0.234
- Freecode 200 80epoch
map=16.483 OP=0.026 OR=0.302 OF1=0.374 CP=0.239 CR=0.091 CF1=0.132
map=18.142 OP=0.033 OR=0.169 OF1=0.273 CP=0.153 CR=0.038 CF1=0.061

##TREC-IS

进行了一些试验：
- Bert微调+多注意力：
（D_LR: 0.1； B_LR: 0.01 60个epoch（50））map=23.271 OP=0.045 OR=0.295 OF1=0.307 CP=0.319 CR=0.160 CF1=0.213
（D_LR: 0.01； B_LR: 0.001 17个epoch）map=24.777 OP=0.111 OR=0.280 OF1=0.328 CP=0.294 CR=0.215 CF1=0.249
- Bert微调+多注意力+GAN：
（D_LR: 0.1； B_LR: 0.001 20个epoch）mmap=24.951 OP=0.049 OR=0.296 OF1=0.323 CP=0.315 CR=0.211 CF1=0.253
（D_LR: 0.1； B_LR: 0.001 33个epoch）map=25.279 OP=0.069 OR=0.319 OF1=0.335 CP=0.292 CR=0.245 CF1=0.266



##RCV2
- Bert微调+多注意力：
map=60.341 OP=0.019 OR=0.728 OF1=0.789 CP=0.539 CR=0.392 CF1=0.454
map=64.564 OP=0.018 OR=0.751 OF1=0.808 CP=0.562 CR=0.413 CF1=0.476

# 0125-（论文记录结果）

##Stack Overflow 2000

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 71285,    ,8019（标签数：228）|Bert微调+MLP       |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|47.5 40.2 47.2|
|---|---|---|---|
|L,U,T: 71285,    ,8019（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|62.0 58.4 59.5|
|---|---|---|---|
|L,U,T: 71285,1600,8019（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|64.5 62.4 61.5|
|---|---|---|---|

##Freecode

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 36798,    ,3905（标签数：124）|Bert微调+MLP      |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|46.0 31.6 37.0|
|---|---|---|---|
|L,U,T: 36798,    ,3905（标签数：124）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|49.5 38.1 40.2|
|---|---|---|---|
|L,U,T: 36798,1600,3905（标签数：124）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|49.9 36.0 46.3|
|---|---|---|---|

##TREC-IS
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 29059,    ,8179（标签数：25）|Bert微调+MLP       |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|30.1 19.0 19.8|
|---|---|---|---|
|L,U,T: 29059,    ,8179（标签数：25）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|30.7 21.3 23.3|
|---|---|---|---|
|L,U,T: 29059,1600,8179（标签数：25）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|37.5 26.8 25.0|
|---|---|---|---|

##AAPD

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 48633,    ,6207（标签数：54）|OCD               | |62.9 51.5 30.7|
|L,U,T: 48633,    ,6207（标签数：54）|OCD               | |62.9 51.5 30.7|
|L,U,T: 48633,    ,6207（标签数：54）|OCD               | |62.9 52.1 31.2|
|---|---|---|---|
|L,U,T: 48633,    ,6207（标签数：54）|Bert微调+MLP       |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|70.1 54.4 58.0 |
|---|---|---|---|
|L,U,T: 48633,    ,6207（标签数：54）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|68.5 50.6 54.7|
|---|---|---|---|
|L,U,T: 48633,1600,6207（标签数：54）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|70.2 54.7 58.8|
|---|---|---|---|


##Stack Overflow 2000-600

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 71285,    ,8019（标签数：228）|Bert微调+MLP      |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|68.4 68.7 72.5|
|---|---|---|---|
|L,U,T: 500,    ,8019（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|69.1 68.5 69.2|
|---|---|---|---|
|L,U,T: 500,1600,8019（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|70.2 69.6 71.2|
|---|---|---|---|

##Stack Overflow 600-

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|L,U,T: 71285,    ,8019（标签数：228）|Bert微调+MLP      |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|60.4 59.1 59.2|
|---|---|---|---|
|L,U,T: 500,    ,8019（标签数：228）|Bert微调+多注意力    |epoch: 60;epoch_step: 50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1, B0.01|69.1 68.5 69.1|
|---|---|---|---|
|L,U,T: 500,1600,8019（标签数：228）|Bert微调+多注意力+GAN|epoch:120;epoch_step:110;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|70.2 69.6 71.2|
|---|---|---|---|
