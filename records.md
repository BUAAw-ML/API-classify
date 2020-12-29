

# 实验结果 -1205

## programmerWeb数据集

### tag频率<200的数据集

|实验环境|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|---|
|614服务器(conda:discrete)|label,unlabel,test:3645,0,1563(split:0.7,未加title_ids)|Bert微调+多注意力|epoch:50;BS:4;OPT:SGD;LR:G0.001,D0.01,B0.1|64.618|
|苏州服务器|label,unlabel,test:3645,0,1563(split:0.7,未加title_ids)|Bert微调+多注意力|epoch:50;BS:4;OPT:SGD;LR:G0.001,D0.01,B0.1|65.03|
|614服务器(conda:discrete)|label,unlabel,test:2604,0,1563(split:0.5,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:40;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|62.620|
|614服务器(conda:discrete)|label,unlabel,test:1562,0,1563(split:0.3,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:50;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|58.845|
|苏州服务器|label,unlabel,test:260,,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|38.292  37.583  38.411  38.042  37.163  35.323  34.915  37.344|
|苏州服务器|label,unlabel,test:260,,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|40.888  39.399  37.532|
|---|---|---|---|---|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和);D训练未使用G处理后的噪音数据|epoch:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|47.369  46.496|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和);D训练未使用G处理后的噪音数据|epoch:50;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|47.315  47.826  47.296|
|苏州服务器|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|51.008  51.994  49.915  51.133|
|苏州服务器|label,unlabel,test:520,3125,1563(tsplit:0.1,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|54.454|
|苏州服务器|label,unlabel,test:1040,2605,1563(split:0.2,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|58.86|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|50.9  47.756|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|51.422  51.364 51.543|
|苏州服务器|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN；Generator设置2层（没提的都为1层）|epoch:50;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|52.123|


另外，进行了其它试验，包括：
- 模型在Gnerator学习率为0.0001，0.01下效果不好;
- Gnerator用了G_feat_match效果不好；
- Gnerator设置3层不好，使用dropout不好；
- 只用label做生成对抗反而不好了

### 全部数据集

|实验环境|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|---|
|苏州服务器|label,unlabel,test:612,0,1226(split:0.05,0.9,未加title_ids)|Bert微调+多注意力|epoch:50;ES:13;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.01|40.378|
|苏州服务器|label,unlabel,test:8579,0,1226(split:0.7,0.9,未加title_ids)|Bert微调+多注意力|epoch:50;ES:13;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.01|59.774|
|苏州服务器|label,unlabel,test:11030,0,1226(split:0.9,未加title_ids)|Bert微调+多注意力|epoch:50;ES:13;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.01|62|
|苏州服务器|label,unlabel,test:612,,1226(split:0.05,0.9,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;ES:45；BS:10;OPT:SGD;LR:G0.001,D0.1,B0.01|44|
|苏州服务器|label,unlabel,test:2448,,1226(split:0.2,0.9,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;ES:45；BS:10;OPT:SGD;LR:G0.001,D0.1,B0.01|51.098|

小结：
- 采用全部数据集（115个标签）时，提出的方法的效果只好大概百分之四，不是很明显；
- programmerWeb数据集性上模型训练到d_loss变为0的时候性能不会下降，还会略微慢慢提升；
- 去掉标注数据的无监督损失不影响最终性能 

### tag频率<100的数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|label,unlabel,test:72,0,438(split:0.05,0.7,未加title_ids)|Bert微调+多注意力|epoch:20;ES:13;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|46.170  45.024  48.917  46.872  43.485  47.755|
|label,unlabel,test:72,949,438(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;ES:45;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|54.745  54.383  55.709  52.864|

另外，进行了其它试验，包括：
- model里的判别特征如何改成和权重矩阵乘后求mean()效果是不好的。
- 尝试了对所有未标注样本打伪标签（预测概率最大的类别tag设置为1），然后一起训练模型。但是基本训练不起来。

## gan-bert数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|label,unlabel,test:109,0,500|Bert微调+多注意力|epoch:20;ES:13;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|20  22  23|
|label,unlabel,test:109,5343,500|Bert微调+多注意力+GAN|epoch:50;ES:45;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|28|


## AAPD数据集
标签数：54
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|label,unlabel,test:548,,3291|Bert微调+多注意力|epoch:50;ES:15;BS:30;OPT:SGD;LR:G0.001,D0.1,B0.01|28.599  28|
|label,unlabel,test:43323,,10968|Bert微调+多注意力|epoch:8;ES:;BS:30;OPT:SGD;LR:G0.001,D0.1,B0.01|58.760|
|label,unlabel,test:49301,,5484|Bert微调+多注意力|epoch:;ES:;BS:30;OPT:SGD;LR:G0.001,D0.1,B0.01|60.641|
|---|---|---|---|
|label,unlabel,test:548,,16452|Bert微调+多注意力|epoch:21;ES:15;BS:30;OPT:SGD;LR:G0.001,D0.1,B0.01|28.793|
|label,unlabel,test:548,37840,16452|Bert微调+多注意力+GAN|epoch:10;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|34.139|
|label,unlabel,test:548,37840,16452|Bert微调+多注意力+GAN|epoch:15;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.001|31.651|
|---|---|---|---|
|label,unlabel,test:548,,3291|Bert微调+多注意力|epoch:50;ES:15;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|34.768|
|label,unlabel,test:548,7129,3291|Bert微调+多注意力+GAN|epoch:23;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|39.414|
|---|---|---|---|
|label,unlabel,test:548,,3291|Bert微调+多注意力|epoch:50;ES:15;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.01|34.129  32.716|
|label,unlabel,test:548,7129,3291|Bert微调+多注意力+GAN|epoch:23;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.001|40.419|
|---|---|---|---|
|label,unlabel,test:4387,,2194|Bert微调+多注意力|epoch:50;ES:15;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.01|48.924  48.842|
|label,unlabel,test:4387,4387,2194|Bert微调+多注意力+GAN|epoch:20;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.001|50.165|
|---|---|---|---|
|label,unlabel,test:7677,,2194|Bert微调+多注意力|epoch:31;ES:13;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.01|52.405  51.908|
|label,unlabel,test:7677,1097,2194|Bert微调+多注意力+GAN|epoch:;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.001|53.727|

另外，进行了其它试验，包括：
- 0.69的label，0.01的。
- batch-size使用30时，GAN初期提不起来（6轮都不咋提高），感觉之后效果应该不好。
- 提出方法当模型达到最高性能后性能又会快速下降（掉到底）（好像是在d_loss变为0的时候）
- 感觉batch-size对方法的效果有影响
- 给generator增加了一层也还是不能避免d_loss变为0后性能迅速下降

## EUR-Lex数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|label,unlabel,test:2176,,1084（标签数：60）（tag频率>200,text_len<510）|Bert微调+多注意力|epoch:22;ES:13;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.01|55.069|
|label,unlabel,test:2176,2177,1084（标签数：60）（tag频率>200,text_len<510）|Bert微调+多注意力+GAN|epoch:45;ES:40;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.001|55.577|
|label,unlabel,test:4353,,1084（标签数：60）（tag频率>200,text_len<510）|Bert微调+多注意力|epoch:45;ES:13;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.01|62.487|
|---|---|---|---|
|label,unlabel,test:5422,,2662（标签数：60）（tag频率>200,text按510截断）|Bert微调+多注意力+GAN|epoch:23;ES:13;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.001|63.559|
|label,unlabel,test:5422,5423,2662（标签数：60）（tag频率>200,text按510截断）|Bert微调+多注意力+GAN|epoch:35;ES:40;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.001|63.904|
|label,unlabel,test:10845,,2662（标签数：60）（tag频率>200,text按510截断）|Bert微调+多注意力|epoch:17;ES:13;BS:10;OPT:SGD;LR:G0.001,D0.1,B0.01|66.914|
|---|---|---|---|
|label,unlabel,test:870,,1084（标签数：60）（tag频率>200,skip）|Bert微调+多注意力+GAN|epoch:30;ES:15;BS:8;OPT:SGD;LR:G0.001,D0.1,B0.001|45.705|
|label,unlabel,test:870,3483,1084（标签数：60）（tag频率>200,skip）|Bert微调+多注意力+GAN|epoch:60;ES:50;BS:8;OPT:SGD;LR:G0.001,D0.1,B0.001|47.850|
|---|---|---|---|
|label,unlabel,test:435，，1084（标签数：60）（tag频率>200,skip）|Bert微调+多注意力+GAN|epoch:30;ES:15;BS:8;OPT:SGD;LR:G0.001,D0.1,B0.001|37.547|
|label,unlabel,test:435，3918，1084（标签数：60）（tag频率>200,skip）|Bert微调+多注意力+GAN|epoch:60;ES:50;BS:8;OPT:SGD;LR:G0.001,D0.1,B0.001|37.876|
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
|label,unlabel,test:800,,132（标签数：41）（tag频率<5000,intanceNum_limit<5000）|Bert微调+多注意力|epoch:20;ES:15;BS:8;OPT:SGD;LR:G0.001,D0.1,B0.01|55.890|
|label,unlabel,test:800,3201,132（标签数：41）（tag频率<5000,intanceNum_limit<5000）|Bert微调+多注意力+GAN|epoch:40;ES:30;BS:8;OPT:SGD;LR:G0.001,D0.1,B0.001|58.535|
|---|---|---|---|
|label,unlabel,test:1000,,410（标签数：41）（tag频率<20000,intanceNum_limit<10000）|Bert微调+多注意力|epoch:25;ES:15;BS:8;OPT:SGD;LR:G0.001,D0.1,B0.01|44.963|
|label,unlabel,test:1000,9001,410（标签数：41）（tag频率<20000,intanceNum_limit<10000）|Bert微调+多注意力+GAN|epoch:40;ES:30;BS:8;OPT:SGD;LR:G0.001,D0.1,B0.001|47.398|

另外进行的试验：
- 提出方法当模型达到最高性能后性能又会快速下降（好像是在d_loss变为0的时候）
- 使用该数据1500，13501，668 提出的方法没有训练成果，具体因为训练中性能掉到底两次

## Stack Overflow数据集
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|label,unlabel,test:500,,262（标签数：205）（tag频率<200,intanceNum_limit<5000）|Bert微调+多注意力|epoch:30;ES:20;BS:8;OPT:SGD;LR:G0.001,D0.1,B0.01|34.023|
|label,unlabel,test:500,4501,262（标签数：205）（tag频率<200,intanceNum_limit<5000）|Bert微调+多注意力+GAN|epoch:75;ES:65;BS:8;OPT:SGD;LR:G0.001,D0.1,B0.001|40.081|


另外进行的试验：
- 采用大规模数据集时，例如tag频率<500，提出方法性能提升到一半就提不动了。
- 在上面第二个实验中，即使当D_loss变为0，模型性能依然能提升。
- 在上面第二个实验中，使用提出方法的修改（fine-grained），性能超过了baseline，但是忽然D_loss变为0，模型性能提升就停滞了
- 学习率、批大小、优化器（Adam训练不起来）都无法解决不能进一步提高的问题
- 把标注数据放在前面训练效果好一点

# 实验结果 1205-1218
## programmerWeb数据集

实验发现：
- 数据量较少时，多注意力后用一个线性层效果好；数据量较多时，多注意力后用分别的权重效果好。
- 采用tag注意力其实就是：计算出每个tag的注意力后，其中最大注意力值若在所有tag里还最大，则该tag很大可能就为预测结果。
- 随着训练进行，tag和token的平均similarity会越来越小（从正到负）
- 把对于真实类别的类别预测概率平均值通过设置loss函数快速抬起来往往导致训练崩溃，因为所有类别的预测概率都被一起抬起来了。

# 实验结果 1218 - 1225           
## programmerWeb数据集 
用所有的unlabel数据集做生成对抗训练
数据配置|模型方法|训练参数|实验结果|                                                                                                                                                                    
|---|---|---|---|                                                                                                                                                                
label,unlabel,test:14,,438（标签数：33）（tag频率<100）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|17.622|             
label,unlabel,test:14,1007,438（标签数：33）（tag频率<100）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|24.300|   
label,unlabel,test:72,,438（标签数：33）（tag频率<100）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|47.951|             
label,unlabel,test:72,949,438（标签数：33）（tag频率<100）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|56.759|    
label,unlabel,test:72,,438（标签数：33）（tag频率<100）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|56.115| 
label,unlabel,test:145,,438（标签数：33）（tag频率<100）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|54.206|  
label,unlabel,test:145,,（标签数：33）（tag频率<100）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|61.973| 
|---|---|---|---| 
label,unlabel,test:52,,1563（标签数：71）（tag频率<200）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|14.254|
label,unlabel,test:52,,1563（标签数：71）（tag频率<200）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|24.505| 
label,unlabel,test:260,,1563（标签数：71）（tag频率<200）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|42|      
label,unlabel,test:260,3385,1563（标签数：71）（tag频率<200）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|50| 
label,unlabel,test:260,,1563（标签数：71）（tag频率<200）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|47.745|  
label,unlabel,test:520,,1563（标签数：71）（tag频率<200）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|50.660|
label,unlabel,test:520,,1563（标签数：71）（tag频率<200）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|55.658| 
|---|---|---|---|
label,unlabel,test:72,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|18.813|
label,unlabel,test:72,4981,2166（标签数：88）（tag频率<300）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|14| 
label,unlabel,test:360,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|42.163|      
label,unlabel,test:360,4693,2166（标签数：88）（tag频率<300）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|45.856|
label,unlabel,test:360,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|45| 
label,unlabel,test:721,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|53.853|    
label,unlabel,test:721,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|53.8029| 
label,unlabel,test:1443,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|58.432|
label,unlabel,test:1443,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|57.989| 
|---|---|---|---|                                                                                                                             
label,unlabel,test:612,,3677（标签数：115）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|45.633|      
label,unlabel,test:612,7967,3677（标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|41.466，41.465|  
|---|---|---|---| 


用是label数据数量3倍的unlabel数据集做生成对抗训练
数据配置|模型方法|训练参数|实验结果|                                                                                                                                                                    
|---|---|---|---|
label,unlabel,test:14,,438（split:0.01,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|20.514|      
label,unlabel,test:14,,438（split:0.01,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|26.960|  
label,unlabel,test:72,,438（split:0.05,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|51.628|      
label,unlabel,test:72,,438（split:0.05,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|52.131|  
label,unlabel,test:145,,438（split:0.1,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|53.741|      
label,unlabel,test:145,,438（split:0.1,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|63.502|
|---|---|---|---|
label,unlabel,test:72,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|18.560|
label,unlabel,test:72,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|19.301| 
label,unlabel,test:721,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|53.449|    
label,unlabel,test:721,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|55.586| 
label,unlabel,test:1443,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|58.669|
label,unlabel,test:1443,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|57.397| 
|---|---|---|---|
label,unlabel,test:122,,3677（标签数：115）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|25.311|      
label,unlabel,test:122,,3677（标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|27.297|  
label,unlabel,test:612,,3677（标签数：115）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|44.263|    
label,unlabel,test:612,,3677（标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|48.640| 
label,unlabel,test:1225,,3677（标签数：115）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|50.900|      
label,unlabel,test:1225,,3677（标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|52.025| 
|---|---|---|---| 


用和label数据数量相等的unlabel数据集做生成对抗训练
数据配置|模型方法|训练参数|实验结果|                                                                                                                                                                    
|---|---|---|---|
label,unlabel,test:14,,438（split:0.01,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|22.850|      
label,unlabel,test:14,,438（split:0.01,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|16.763|  
label,unlabel,test:72,,438（split:0.05,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|51.504|      
label,unlabel,test:72,,438（split:0.05,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|50.239|  
label,unlabel,test:145,,438（split:0.1,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|64.491|      
label,unlabel,test:145,,438（split:0.1,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|61.855|
label,unlabel,test:,,438（split:0.15,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01||      
label,unlabel,test:,,438（split:0.15,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||  
label,unlabel,test:,,438（split:0.2,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01||      
label,unlabel,test:,,438（split:0.2,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||  
label,unlabel,test:,,438（split:0.35,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01||      
label,unlabel,test:,,438（split:0.35,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||  
|---|---|---|---|
label,unlabel,test:52,,1563（split:0.01,标签数：71）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|13.052|      
label,unlabel,test:52,,1563（split:0.01,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|11.664|  
label,unlabel,test:260,,1563（split:0.05,标签数：71）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|44.632|      
label,unlabel,test:260,,1563（split:0.05,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|48.650| 
label,unlabel,test:520,,1563（split:0.1,标签数：71）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|51.797|      
label,unlabel,test:520,,1563（split:0.1,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|54.396|
label,unlabel,test:781,,1563（split:0.01,标签数：71）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|53.554|      
label,unlabel,test:781,,1563（split:0.01,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|56.755| 
label,unlabel,test:1041,,1563（split:0.01,标签数：71）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|58.404|      
label,unlabel,test:1041,,1563（split:0.01,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|59.463|
label,unlabel,test:1822,,1563（split:0.01,标签数：71）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|61.551|      
label,unlabel,test:1822,,1563（split:0.01,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|62.476|   
label,unlabel,test:3645,,1563（split:0.7,标签数：71）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|66.331 66.077| 
|---|---|---|---|
label,unlabel,test:72,,2166（split:0.01,标签数：88）（tag频率<300）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|16.873|
label,unlabel,test:72,,2166（split:0.01,标签数：88）（tag频率<300）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|16.866| 
label,unlabel,test:721,,2166（split:0.1,标签数：88）（tag频率<300）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|54.027|    
label,unlabel,test:721,,2166（split:0.1,标签数：88）（tag频率<300）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|55.879| 
label,unlabel,test:1443,,2166（split:0.2,标签数：88）（tag频率<300）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|58.031|
label,unlabel,test:1443,,2166（split:0.2,标签数：88）（tag频率<300）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|59.036| 
label,unlabel,test:5053,,2166（split:0.7,标签数：88）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|55.055 63.592|
|---|---|---|---|                                                                                                                             
label,unlabel,test:122,,3677（split:0.01,标签数：115）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|24.649|      
label,unlabel,test:122,,3677（split:0.01,标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|26.872|  
label,unlabel,test:612,,3677（split:0.05,标签数：115）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|45.749|      
label,unlabel,test:612,,3677（split:0.05,标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|48.214| 
label,unlabel,test:1225,,3677（split:0.1,标签数：115）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|52.079|      
label,unlabel,test:1225,,3677（split:0.1,标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|52.748| 
label,unlabel,test:1838,,3677（split:0.15,标签数：115）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|54.369|      
label,unlabel,test:1838,,3677（split:0.15,标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|55.065| 
label,unlabel,test:2451,,3677（split:0.2,标签数：115）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|56.226|      
label,unlabel,test:2451,,3677（split:0.2,标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|56.279| 
label,unlabel,test:4289,,3677（split:0.35,标签数：115）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|58.470|      
label,unlabel,test:4289,,3677（split:0.35,标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|58.894| 
label,unlabel,test:8579,,3677（split:0.7,标签数：115）|Bert微调+多注意力|epoch:50;ES:20;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|61.771 61.648|
|---|---|---|---| 


用是label数据数量5倍的unlabel数据集做生成对抗训练
数据配置|模型方法|训练参数|实验结果|                                                                                                                                                                    
|---|---|---|---|
label,unlabel,test:14,,438（split:0.01,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|19.885|      
label,unlabel,test:14,,438（split:0.01,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|21.677|  
label,unlabel,test:72,,438（split:0.05,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|51.869|      
label,unlabel,test:72,,438（split:0.05,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|56.104|  
label,unlabel,test:145,,438（split:0.1,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01||      
label,unlabel,test:145,,438（split:0.1,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||
|---|---|---|---|

用最大训练数据量做生成对抗训练
数据配置|模型方法|训练参数|实验结果|                                                                                                                                                                    
|---|---|---|---|
label,unlabel,test:,,（split:,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01||      
label,unlabel,test:,,（split:,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||  
label,unlabel,test:3645,,1563（split:,标签数：71）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|66.254|      
label,unlabel,test:3645,3645,1563（split:,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|65.096|  
label,unlabel,test:8579,,3677（split:,标签数：115）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|61.546|      
label,unlabel,test:8579,8579,3677（split:,标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|60.510|
|---|---|---|---|

用1000数据量做生成对抗训练
数据配置|模型方法|训练参数|实验结果| 
|---|---|---|---|
label,unlabel,test:14,,438（split:0.01,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|17.166|      
label,unlabel,test:14,1000,438（split:0.01,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|29.159|  
label,unlabel,test:72,,438（split:0.05,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|51.192|      
label,unlabel,test:72,1000,438（split:0.05,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|55.538|  
label,unlabel,test:145,,438（split:0.1,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|58.517|      
label,unlabel,test:145,1000,438（split:0.1,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|65.665|   
label,unlabel,test:218,,438（split:0.15,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|69.581|      
label,unlabel,test:218,1000,438（split:0.15,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|70.604|  
label,unlabel,test:291,,438（split:0.2,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|71.984|      
label,unlabel,test:291,1000,438（split:0.2,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|73.270|  
label,unlabel,test:510,,438（split:0.35,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|72.784|      
label,unlabel,test:510,1000,438（split:0.35,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|74.837|
label,unlabel,test:656,,438（split:0.45,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|76.589|      
label,unlabel,test:656,1000,438（split:0.45,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|77.144|  
label,unlabel,test:802,,438（split:0.55,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|77.129|      
label,unlabel,test:802,1000,438（split:0.55,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|78.063|  
label,unlabel,test:1021,,438（split:0.7,标签数：33）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|79.664|      
label,unlabel,test:1021,1000,438（split:0.7,标签数：33）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|78.949|                                                                                                                                                                      
|---|---|---|---|
label,unlabel,test:52,,1563（split:0.01,标签数：71）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|14.747|      
label,unlabel,test:52,1000,1563（split:0.01,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|20.352|  
label,unlabel,test:260,,1563（split:0.05,标签数：71）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|43.274|      
label,unlabel,test:260,1000,1563（split:0.05,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|51.043|  
label,unlabel,test:520,,1563（split:0.1,标签数：71）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|51.142|      
label,unlabel,test:520,1000,1563（split:0.1,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|54.112|  
label,unlabel,test:781,,1563（split:0.15,标签数：71）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|54.711|      
label,unlabel,test:781,1000,1563（split:0.15,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|56.464|  
label,unlabel,test:1041,,1563（split:0.2,标签数：71）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|57.366|      
label,unlabel,test:1041,1000,1563（split:0.2,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|58.463| 
label,unlabel,test:1302,,1563（split:0.25,标签数：71）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|59.828|      
label,unlabel,test:1302,1000,1563（split:0.25,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|59.780|  
label,unlabel,test:1822,,1563（split:0.35,标签数：71）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|61.602|      
label,unlabel,test:1822,1000,1563（split:0.35,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|62.616|  
label,unlabel,test:2343,,1563（split:0.45,标签数：71）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|62.160|      
label,unlabel,test:2343,1000,1563（split:0.45,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|63.060|  
label,unlabel,test:2864,,1563（split:0.55,标签数：71）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|65.241|      
label,unlabel,test:2864,1000,1563（split:0.55,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|64.227|  
label,unlabel,test:3645,,1563（split:0.7,标签数：71）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|65.363|      
label,unlabel,test:3645,1000,1563（split:0.7,标签数：71）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|65.267|  
|---|---|---|---|
label,unlabel,test:52,,1563（split:0.01,标签数：71）|Bert微调+多注意力+直接相加|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|37.919| 
label,unlabel,test:260,,1563（split:0.05,标签数：71）|Bert微调+多注意力+直接相加|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|51.745| 
label,unlabel,test:520,,1563（split:0.1,标签数：71）|Bert微调+多注意力+直接相加|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|55.748| 
label,unlabel,test:781,,1563（split:0.15,标签数：71）|Bert微调+多注意力+直接相加|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|56.572|  
label,unlabel,test:1041,,1563（split:0.2,标签数：71）|Bert微调+多注意力+直接相加|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|58.602|     
label,unlabel,test:1302,,1563（split:0.25,标签数：71）|Bert微调+多注意力+直接相加|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|59.384|
label,unlabel,test:1822,,1563（split:0.35,标签数：71）|Bert微调+多注意力+直接相加|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|61.955| 
label,unlabel,test:2343,,1563（split:0.45,标签数：71）|Bert微调+多注意力+直接相加|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|63.977|  
label,unlabel,test:2864,,1563（split:0.55,标签数：71）|Bert微调+多注意力+直接相加|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|63.878|     
label,unlabel,test:3645,,1563（split:0.7,标签数：71）|Bert微调+多注意力+直接相加|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|65.299| 
|---|---|---|---|
label,unlabel,test:52,,1563（split:0.01,标签数：71）|Bert微调+多注意力+Linear|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|39.750| 
label,unlabel,test:781,,1563（split:0.15,标签数：71）|Bert微调+多注意力+Linear|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|56.733|  
label,unlabel,test:1822,,1563（split:0.35,标签数：71）|Bert微调+多注意力+Linear|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|61.379|    
label,unlabel,test:3645,,1563（split:0.7,标签数：71）|Bert微调+多注意力+Linear|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|64.077| 
|---|---|---|---|
label,unlabel,test:122,,3677（split:0.01,标签数：115）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|27.207|      
label,unlabel,test:122,1000,3677（split:0.01,标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|33.133|  
label,unlabel,test:,,（split:0.05,标签数：115）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01||      
label,unlabel,test:,,（split:0.05,标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||  
label,unlabel,test:612,,3677（split:0.1,标签数：115）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|44.704|      
label,unlabel,test:612,1000,3677（split:0.1,标签数：115）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|49.585|
|---|---|---|---|

# 实验结果 1226 - 1227
## programmerWeb数据集 （全集473个tag）

数据配置|模型方法|训练参数|实验结果|                                                                                                                                                                    
|---|---|---|---|     
label,unlabel,test:645,,2584（split:0.05,标签数：473）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|32.499|
label,unlabel,test:645,1000,2584（split:0.05,标签数：473）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|32.713|
label,unlabel,test:645,500,2584（split:0.05,标签数：473）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|34.307|
label,unlabel,test:1291,,2584（split:0.1,标签数：473）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|38.963|
label,unlabel,test:1291,1000,2584（split:0.1,标签数：473）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|36.319|
label,unlabel,test:1291,500,2584（split:0.1,标签数：473）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001| 37.402|
label,unlabel,test:2583,,2584（split:0.2,标签数：473）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|43.523| 
label,unlabel,test:2583,1000,2584（split:0.2,标签数：473）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|39.260|
label,unlabel,test:2583,500,2584（split:0.2,标签数：473）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|39.844|
label,unlabel,test:3875,,2584（split:0.3,标签数：473）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|43.523|      
label,unlabel,test:3875,1000,2584（split:0.3,标签数：473）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||
|---|---|---|---|

-用3000的生成对抗训练更不好


过滤频率100以下的tag
数据配置|模型方法|训练参数|实验结果|                                                                                                                                                                    
|---|---|---|---|     
label,unlabel,test:,,（split:0.05,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|47.136|
label,unlabel,test:,3000,（split:0.05,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|49.122|
label,unlabel,test:,3000,（split:0.05,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|50.169|
label,unlabel,test:,,（split:0.1,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|52.026|
label,unlabel,test:,3000,（split:0.1,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|52.412|
label,unlabel,test:,500,（split:0.1,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|53.728|
label,unlabel,test:2555,,（split:0.2,标签数：）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|| 
label,unlabel,test:2555,3000,（split:0.2,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|54.772|
label,unlabel,test:2555,500,（split:0.2,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|56.759|
label,unlabel,test:,,（split:0.3,标签数：）|Bert微调+多注意力|epoch:50;ES:30;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.01|56.371|      
label,unlabel,test:,3000,（split:0.3,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||
label,unlabel,test:,500,（split:0.3,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|57.647|
|---|---|---|---|

过滤频率50以下的tag
数据配置|模型方法|训练参数|实验结果|                                                                                                                                                                    
|---|---|---|---|   
label,unlabel,test:644,,2579（split:0.05,标签数：244）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|38.875|
label,unlabel,test:644,500,2579（split:0.05,标签数：244）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|41.994|
label,unlabel,test:1289,,2579（split:0.1,标签数：244）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|44.329|
label,unlabel,test:1289,500,2579（split:0.1,标签数：244）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|45.210|
label,unlabel,test:2579,,2579（split:0.1,标签数：244）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|47.099|
label,unlabel,test:2579,500,2579（split:0.1,标签数：244）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|47.762|
label,unlabel,test:,,2579（split:0.2,标签数：244）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||
label,unlabel,test:,500,2579（split:0.2,标签数：244）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||
|---|---|---|---|   

# 实验结果 1226 - 1227
## programmerWeb数据集全集按照8：2对每个tag样本分割，过滤掉只有一两个样本的tag，剩406个tag
此处的split只针对train数据集
数据配置|模型方法|训练参数|实验结果|                                                                                                                                                                    
|---|---|---|---|   
label,unlabel,test:513,,2579（split:0.05,标签数：）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|22.753 23.298|
label,unlabel,test:513,1000,2579（split:0.05,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|26.448 28.282|
label,unlabel,test:513,500,2579（split:0.05,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|26.448|
label,unlabel,test:1026,,2579（split:0.1,标签数：）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|30.056 31.188  30.420|
label,unlabel,test:1026,1000,2579（split:0.1,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|30.793 31.148|
label,unlabel,test:2052,,2579（split:0.2,标签数：）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|35.568 35.340|
label,unlabel,test:2052,1000,2579（split:0.2,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|34.371|
label,unlabel,test:,,2579（split:,标签数：）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||
label,unlabel,test:,1000,2579（split:,标签数：）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||
label,unlabel,test:4104,,2659（split:0.4,标签数：406）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|37.838|
label,unlabel,test:4104,500,2659（split:0.4,标签数：406）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|35.172|
|---|---|---|---|   

过滤tag频率小于50的
数据配置|模型方法|训练参数|实验结果|                                                                                                                                                                    
|---|---|---|---|   
label,unlabel,test:490,,2544（split:0.05,标签数：112）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|34.223|
label,unlabel,test:490,500,2544（split:0.05,标签数：112）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|38.627|
label,unlabel,test:980,,2544（split:0.1,标签数：112）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|40.215|
label,unlabel,test:980,500,2544（split:0.1,标签数：112）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|44.202|
label,unlabel,test:1960,,2544（split:0.2,标签数：112）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|43.912|
label,unlabel,test:1960,500,2544（split:0.2,标签数：112）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|46.125|

过滤tag频率小于20的
数据配置|模型方法|训练参数|实验结果|                                                                                                                                                                  
|---|---|---|---|   
label,unlabel,test:507,,2626（split:0.05,标签数：213）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|27.177|
label,unlabel,test:507,500,2626（split:0.05,标签数：213）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|32.456|
label,unlabel,test:1521,,2626（split:0.15,标签数：213）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|36.887|
label,unlabel,test:1521,500,2626（split:0.15,标签数：213）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|37.111|
label,unlabel,test:2028,,2626（split:0.2,标签数：213）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|39.104|
label,unlabel,test:2028,500,2626（split:0.2,标签数：213）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|39.344|
label,unlabel,test:4056,,2626（split:0.4,标签数：213）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|42.853|
label,unlabel,test:4056,500,2626（split:0.4,标签数：213）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|42.111|
label,unlabel,test:,,2626（split:0.,标签数：213）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||
label,unlabel,test:8112,500,2626（split:0.,标签数：213）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|44.585|

过滤tag频率小于10的
数据配置|模型方法|训练参数|实验结果|            wqb                                                                                                                                                         
|---|---|---|---|   
label,unlabel,test:510,,2648（split:0.05,标签数：291）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|25.378|
label,unlabel,test:510,500,2648（split:0.05,标签数：291）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|30.525|
label,unlabel,test:1021,,2648（split:0.1,标签数：291）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|32.359|
label,unlabel,test:1021,500,2648（split:0.1,标签数：291）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|33.743|
label,unlabel,test:2043,,2648（split:0.2,标签数：291）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|37.202|
label,unlabel,test:2043,500,2648（split:0.2,标签数：291）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|36.678|
label,unlabel,test:4087,,2648（split:0.4,标签数：291）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|40.399|
label,unlabel,test:4087,500,2648（split:0.4,标签数：291）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||
label,unlabel,test:,,2648（split:0.,标签数：291）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|41.924|
label,unlabel,test:,500,2648（split:0.,标签数：291）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|40.375|

过滤tag频率小于5的
数据配置|模型方法|训练参数|实验结果|            zyc                                                                                                                                                         
|---|---|---|---|   
label,unlabel,test:512,,2656（split:0.05,标签数：370）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|23.833|
label,unlabel,test:512,500,2656（split:0.05,标签数：370）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|29.188|
label,unlabel,test:1025,,2656（split:0.1,标签数：370）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|29.922|
label,unlabel,test:1025,500,2656（split:0.1,标签数：370）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||
label,unlabel,test:2051,,2656（split:0.2,标签数：370）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||
label,unlabel,test:2051,500,2656（split:0.2,标签数：370）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|34.578|
label,unlabel,test:,,2656（split:0.4,标签数：370）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|39.376|
label,unlabel,test:,500,2656（split:0.4,标签数：370）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001||
label,unlabel,test:,,2656（split:0.8,标签数：370）|Bert微调+多注意力|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|40.296|
label,unlabel,test:,500,2656（split:0.8,标签数：370）|Bert微调+多注意力+GAN|epoch:100;ES:90;BS:4;OPT:SGD;LR:G0.001,D0.1,B0.001|38.005|