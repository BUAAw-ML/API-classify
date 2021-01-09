
# Experiment runner script

#EN='02'  #experiment_no

python main.py \
        --experiment_no='wqb_alldata_0<tag<50'  \
        --epochs=50 \
        --epoch_step=40 \
        --device_ids=0 \
        --batch-size=4 \
        --G-lr=0.001 \
        --D-lr=0.1 \
        --B-lr=0.01 \
        --save_model_path='./checkpoint' \
        --data_type='TrainTest' \
        --data_path='../datasets/ProgrammerWeb/' \
        --use_previousData=0 \
        --method='MultiLabelMAP' \
        --overlength_handle='truncation' \
        --min_tagFrequence=0  \
        --max_tagFrequence=150 \
        --intanceNum_limit=99999 \
        --data_split=99999 \
        --test_description=''  \
        --model_backend='pytorch'

#python main.py \
#        --experiment_no='wqb_alldata_0<tag<50'  \
#        --epochs=100 \
#        --epoch_step=90 \
#        --device_ids=0 \
#        --batch-size=4 \
#        --G-lr=0.001 \
#        --D-lr=0.1 \
#        --B-lr=0.001 \
#        --save_model_path='./checkpoint' \
#        --data_type='TrainTest' \
#        --data_path='../datasets/ProgrammerWeb/' \
#        --use_previousData=0 \
#        --method='semiGAN_MultiLabelMAP' \
#        --overlength_handle='truncation' \
#        --min_tagFrequence=0  \
#        --max_tagFrequence=50 \
#        --intanceNum_limit=99999 \
#        --data_split=99999 \
#        --test_description=''  \
#        --model_backend='pytorch'


#方法、epoch_step

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

#../datasets/ag-news

#614服务器上数据：
#../../datasets/multiClass_text_classification/news_group20/news_group20.csv
#../../datasets/multiLabel_text_classification/ProgrammerWeb/programweb-data.csv
#../../datasets/multiLabel_text_classification/EUR-Lex


