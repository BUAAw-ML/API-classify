

# 实验结果 -1205

## programmerWeb数据集

### tag频率<200的数据集

|实验环境|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|---|
|614服务器(conda:discrete)|label,unlabel,test:3645,0,1563(split:0.7,未加title_ids)|Bert微调+多注意力|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.01,B0.1|64.618|
|苏州服务器|label,unlabel,test:3645,0,1563(split:0.7,未加title_ids)|Bert微调+多注意力|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.01,B0.1|65.03|
|614服务器(conda:discrete)|label,unlabel,test:2604,0,1563(split:0.5,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:40;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|62.620|
|614服务器(conda:discrete)|label,unlabel,test:1562,0,1563(split:0.3,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|58.845|
|苏州服务器|label,unlabel,test:260,,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|38.292  37.583  38.411  38.042  37.163  35.323  34.915  37.344|
|苏州服务器|label,unlabel,test:260,,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力;未使用unlabel训练数据|epoch:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|40.888  39.399  37.532|
|---|---|---|---|---|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和);D训练未使用G处理后的噪音数据|epoch:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.369  46.496|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN（对抗指标：噪音样本与标签w近似度的和);D训练未使用G处理后的噪音数据|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.315  47.826  47.296|
|苏州服务器|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|51.008  51.994  49.915  51.133|
|苏州服务器|label,unlabel,test:520,3125,1563(tsplit:0.1,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|54.454|
|苏州服务器|label,unlabel,test:1040,2605,1563(split:0.2,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|58.86|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|50.9  47.756|
|614服务器(conda:discrete)|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|51.422  51.364 51.543|
|苏州服务器|label,unlabel,test:260,3385,1563(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN；Generator设置2层（没提的都为1层）|epoch:50;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|52.123|


另外，进行了其它试验，包括：
- 模型在Gnerator学习率为0.0001，0.01下效果不好;
- Gnerator用了G_feat_match效果不好；
- Gnerator设置3层不好，使用dropout不好；
- 只用label做生成对抗反而不好了

### 全部数据集

|实验环境|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|---|
|苏州服务器|label,unlabel,test:612,0,1226(split:0.05,0.9,未加title_ids)|Bert微调+多注意力|epoch:50;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|40.378|
|苏州服务器|label,unlabel,test:8579,0,1226(split:0.7,0.9,未加title_ids)|Bert微调+多注意力|epoch:50;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|59.774|
|苏州服务器|label,unlabel,test:11030,0,1226(split:0.9,未加title_ids)|Bert微调+多注意力|epoch:50;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|62|
|苏州服务器|label,unlabel,test:612,,1226(split:0.05,0.9,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;epoch_step:45；batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|44|
|苏州服务器|label,unlabel,test:2448,,1226(split:0.2,0.9,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;epoch_step:45；batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|51.098|

小结：
- 采用全部数据集（115个标签）时，提出的方法的效果只好大概百分之四，不是很明显；
- programmerWeb数据集性上模型训练到d_loss变为0的时候性能不会下降，还会略微慢慢提升；
- 去掉标注数据的无监督损失不影响最终性能 

### tag频率<100的数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|label,unlabel,test:72,0,438(split:0.05,0.7,未加title_ids)|Bert微调+多注意力|epoch:20;epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|46.170  45.024  48.917  46.872  43.485  47.755|
|label,unlabel,test:72,949,438(split:0.05,0.7,未加title_ids)|Bert微调+多注意力+GAN|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|54.745  54.383  55.709  52.864|

另外，进行了其它试验，包括：
- model里的判别特征如何改成和权重矩阵乘后求mean()效果是不好的。
- 尝试了对所有未标注样本打伪标签（预测概率最大的类别tag设置为1），然后一起训练模型。但是基本训练不起来。

## gan-bert数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|label,unlabel,test:109,0,500|Bert微调+多注意力|epoch:20;epoch_step:13;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|20  22  23|
|label,unlabel,test:109,5343,500|Bert微调+多注意力+GAN|epoch:50;epoch_step:45;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|28|


## AAPD数据集
标签数：54
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|label,unlabel,test:548,,3291|Bert微调+多注意力|epoch:50;epoch_step:15;batch-size:30;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|28.599  28|
|label,unlabel,test:43323,,10968|Bert微调+多注意力|epoch:8;epoch_step:;batch-size:30;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|58.760|
|label,unlabel,test:49301,,5484|Bert微调+多注意力|epoch:;epoch_step:;batch-size:30;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|60.641|
|---|---|---|---|
|label,unlabel,test:548,,16452|Bert微调+多注意力|epoch:21;epoch_step:15;batch-size:30;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|28.793|
|label,unlabel,test:548,37840,16452|Bert微调+多注意力+GAN|epoch:10;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|34.139|
|label,unlabel,test:548,37840,16452|Bert微调+多注意力+GAN|epoch:15;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|31.651|
|---|---|---|---|
|label,unlabel,test:548,,3291|Bert微调+多注意力|epoch:50;epoch_step:15;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|34.768|
|label,unlabel,test:548,7129,3291|Bert微调+多注意力+GAN|epoch:23;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|39.414|
|---|---|---|---|
|label,unlabel,test:548,,3291|Bert微调+多注意力|epoch:50;epoch_step:15;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|34.129  32.716|
|label,unlabel,test:548,7129,3291|Bert微调+多注意力+GAN|epoch:23;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|40.419|
|---|---|---|---|
|label,unlabel,test:4387,,2194|Bert微调+多注意力|epoch:50;epoch_step:15;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|48.924  48.842|
|label,unlabel,test:4387,4387,2194|Bert微调+多注意力+GAN|epoch:20;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|50.165|
|---|---|---|---|
|label,unlabel,test:7677,,2194|Bert微调+多注意力|epoch:31;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|52.405  51.908|
|label,unlabel,test:7677,1097,2194|Bert微调+多注意力+GAN|epoch:;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|53.727|

另外，进行了其它试验，包括：
- 0.69的label，0.01的。
- batch-size使用30时，GAN初期提不起来（6轮都不咋提高），感觉之后效果应该不好。
- 提出方法当模型达到最高性能后性能又会快速下降（掉到底）（好像是在d_loss变为0的时候）
- 感觉batch-size对方法的效果有影响
- 给generator增加了一层也还是不能避免d_loss变为0后性能迅速下降

## EUR-Lex数据集

|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|label,unlabel,test:2176,,1084（标签数：60）（tag频率>200,text_len<510）|Bert微调+多注意力|epoch:22;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|55.069|
|label,unlabel,test:2176,2177,1084（标签数：60）（tag频率>200,text_len<510）|Bert微调+多注意力+GAN|epoch:45;epoch_step:40;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|55.577|
|label,unlabel,test:4353,,1084（标签数：60）（tag频率>200,text_len<510）|Bert微调+多注意力|epoch:45;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|62.487|
|---|---|---|---|
|label,unlabel,test:5422,,2662（标签数：60）（tag频率>200,text按510截断）|Bert微调+多注意力+GAN|epoch:23;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|63.559|
|label,unlabel,test:5422,5423,2662（标签数：60）（tag频率>200,text按510截断）|Bert微调+多注意力+GAN|epoch:35;epoch_step:40;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|63.904|
|label,unlabel,test:10845,,2662（标签数：60）（tag频率>200,text按510截断）|Bert微调+多注意力|epoch:17;epoch_step:13;batch-size:10;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|66.914|
|---|---|---|---|
|label,unlabel,test:870,,1084（标签数：60）（tag频率>200,skip）|Bert微调+多注意力+GAN|epoch:30;epoch_step:15;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|45.705|
|label,unlabel,test:870,3483,1084（标签数：60）（tag频率>200,skip）|Bert微调+多注意力+GAN|epoch:60;epoch_step:50;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.850|
|---|---|---|---|
|label,unlabel,test:435，，1084（标签数：60）（tag频率>200,skip）|Bert微调+多注意力+GAN|epoch:30;epoch_step:15;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|37.547|
|label,unlabel,test:435，3918，1084（标签数：60）（tag频率>200,skip）|Bert微调+多注意力+GAN|epoch:60;epoch_step:50;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|37.876|
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
|label,unlabel,test:800,,132（标签数：41）（tag频率<5000,intanceNum_limit<5000）|Bert微调+多注意力|epoch:20;epoch_step:15;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|55.890|
|label,unlabel,test:800,3201,132（标签数：41）（tag频率<5000,intanceNum_limit<5000）|Bert微调+多注意力+GAN|epoch:40;epoch_step:30;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|58.535|
|---|---|---|---|
|label,unlabel,test:1000,,410（标签数：41）（tag频率<20000,intanceNum_limit<10000）|Bert微调+多注意力|epoch:25;epoch_step:15;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|44.963|
|label,unlabel,test:1000,9001,410（标签数：41）（tag频率<20000,intanceNum_limit<10000）|Bert微调+多注意力+GAN|epoch:40;epoch_step:30;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.398|

另外进行的试验：
- 提出方法当模型达到最高性能后性能又会快速下降（好像是在d_loss变为0的时候）
- 使用该数据1500，13501，668 提出的方法没有训练成果，具体因为训练中性能掉到底两次

## Stack Overflow数据集
|数据配置|模型方法|训练参数|实验结果|
|---|---|---|---|
|label,unlabel,test:500,,262（标签数：205）（tag频率<200,intanceNum_limit<5000）|Bert微调+多注意力|epoch:30;epoch_step:20;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|34.023|
|label,unlabel,test:500,4501,262（标签数：205）（tag频率<200,intanceNum_limit<5000）|Bert微调+多注意力+GAN|epoch:75;epoch_step:65;batch-size:8;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|40.081|


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

# 实验结果 1218 -            
## programmerWeb数据集   
数据配置|模型方法|训练参数|实验结果|                                                                                                                                                                    
|---|---|---|---|                                                                                                                                                                        
label,unlabel,test:72,,438（标签数：33）（tag频率<100）|Bert微调+多注意力|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|47.951|             
label,unlabel,test:72,949,438（标签数：33）（tag频率<100）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|56.759|    
label,unlabel,test:72,,438（标签数：33）（tag频率<100）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|56.115| 
|---|---|---|---| 
label,unlabel,test:260,,1563（标签数：71）（tag频率<200）|Bert微调+多注意力|epoch:50;epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|42|      
label,unlabel,test:260,3385,1563（标签数：71）（tag频率<200）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|50| 
label,unlabel,test:72,,438（标签数：33）（tag频率<100）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|47.745|  
|---|---|---|---|
label,unlabel,test:360,,2166（标签数：88）（tag频率<300）|Bert微调+多注意力|epoch:50;epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|42.163|      
label,unlabel,test:360,4693,2166（标签数：88）（tag频率<300）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|45.856|
label,unlabel,test:72,,438（标签数：33）（tag频率<100）|Bert微调+多注意力+GAN（不用unlabel）|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|45| 
|---|---|---|---|                                                                                                                             
label,unlabel,test:612,,3677（标签数：115）|Bert微调+多注意力|epoch:50;epoch_step:20;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.01|45.633|      
label,unlabel,test:612,7967,3677（标签数：115）|Bert微调+多注意力+GAN|epoch:100;epoch_step:90;batch-size:4;optimizer:SGD;learning-rate:G0.001,D0.1,B0.001|41.466，41.465|  
|---|---|---|---| 