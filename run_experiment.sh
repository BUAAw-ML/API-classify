
# Experiment runner script

#EN='02'  #experiment_no


python main.py \
        --experiment_no='max_tagFrequence<200_intanceNum_limit=10000_Test'  \
        --epochs=90 \
        --epoch_step=75 \
        --device_ids=0 \
        --batch-size=8 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.001 \
        --save_model_path='./checkpoint' \
        --data_type='TrainTestTextTag' \
        --data_path='../datasets/stack-overflow' \
        --use_previousData=0 \
        --method='semiGAN_MultiLabelMAP' \
        --overlength_handle='skip' \
        --min_tagFrequence=0  \
        --max_tagFrequence=200  \
        --intanceNum_limit=10000 \
        --data_split=0.1  \

#python main.py \
#        --experiment_no='test_tagFrequence<100_data_split=0.05'  \
#        --epochs=60 \
#        --epoch_step=50 \
#        --device_ids=0 \
#        --batch-size=4 \
#        --G-lr=0.001 \
#        --D-lr=0.1 \
#        --B-lr=0.001 \
#        --save_model_path='./checkpoint' \
#        --data_type='All' \
#        --data_path='../datasets/ProgrammerWeb/programweb-data.csv' \
#        --use_previousData=0 \
#        --method='semiGAN_MultiLabelMAP' \
#        --overlength_handle='skip' \
#        --min_tagFrequence=0  \
#        --max_tagFrequence=100  \
#        --intanceNum_limit=10000 \
#        --data_split=0.05  \

#方法、学习率、epoch_step

#batch-size：1，4，8，16
#data_type: All  TrainTest  TrainTestTextTag
#method: MultiLabelMAP semiGAN_MultiLabelMAP
#overlength_handle: truncation  skip

#苏州服务器上数据：
#../datasets/ProgrammerWeb/programweb-data.csv
#../datasets/AAPD/aapd2.csv
#../datasets/gan-bert
#../datasets/EUR-Lex  (TrainTestTextTag)
#../datasets/RCV2  (TrainTestTextTag)
#../datasets/stack-overflow  (TrainTestTextTag)

#614服务器上数据：
#../../datasets/multiClass_text_classification/news_group20/news_group20.csv
#../../datasets/multiLabel_text_classification/ProgrammerWeb/programweb-data.csv
#../../datasets/multiLabel_text_classification/EUR-Lex


