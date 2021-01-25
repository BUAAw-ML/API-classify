
# Experiment runner script

#EN='02'  #experiment_no


#python main.py \
#        --experiment_no='wqb_allData'  \
#        --epochs=50 \
#        --epoch_step=40 \
#        --device_ids=0 \
#        --batch-size=4 \
#        --G-lr=0.001 \
#        --D-lr=0.1 \
#        --B-lr=0.01 \
#        --save_model_path='./checkpoint' \
#        --data_type='TrainTest_programWeb_freecode_AAPD' \
#        --data_path='../datasets/AAPD' \
#        --use_previousData=0 \
#        --method='MultiLabelMAP' \
#        --overlength_handle='truncation' \
#        --min_tagFrequence=0  \
#        --max_tagFrequence=99999  \
#        --intanceNum_limit=99999 \
#        --data_split=99999  \
#        --test_description=''  \

python main.py \
        --experiment_no='wqb_MABert'  \
        --epochs=20 \
        --epoch_step=17 \
        --device_ids=0 \
        --batch-size=4 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.01 \
        --save_model_path='./checkpoint' \
        --data_type='TrainTestTextTag' \
        --data_path='../datasets/RCV2' \
        --use_previousData=0 \
        --model_type='MABert' \
        --method='MultiLabelMAP' \
        --overlength_handle='truncation' \
        --min_tagFrequence=0  \
        --max_tagFrequence=999999  \
        --intanceNum_limit=999999 \
        --data_split=999999  \
        --test_description=''  \

python main.py \
        --experiment_no='wqb_MABert'  \
        --epochs=60 \
        --epoch_step=55 \
        --device_ids=0 \
        --batch-size=4 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.001 \
        --save_model_path='./checkpoint' \
        --data_type='TrainTestTextTag' \
        --data_path='../datasets/RCV2' \
        --use_previousData=0 \
        --model_type='MABert' \
        --method='semiGAN_MultiLabelMAP' \
        --overlength_handle='truncation' \
        --min_tagFrequence=0  \
        --max_tagFrequence=999999  \
        --intanceNum_limit=999999 \
        --data_split=999999  \
        --test_description=''  \

#方法、epoch_step
# MLPBert, MABert
#batch-size：1，4，8，16
#data_type: All  TrainTest  TrainTestTextTag
#method: MultiLabelMAP semiGAN_MultiLabelMAP
#overlength_handle: truncation  skip

#苏州服务器上数据：TrainTest_programWeb_freecode_AAPD
#../datasets/AAPD
#../datasets/Freecode
#../datasets/TREC-IS

#../datasets/EUR-Lex  (TrainTestTextTag)
#../datasets/RCV2  (TrainTestTextTag)
#../datasets/stack-overflow2000 (TrainTestTextTag)

#../datasets/ag-news

#614服务器上数据：
#../../datasets/multiClass_text_classification/news_group20/news_group20.csv
#../../datasets/multiLabel_text_classification/ProgrammerWeb/programweb-data.csv
#../../datasets/multiLabel_text_classification/EUR-Lex


