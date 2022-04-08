import os

modelPath = "result"
dataPath = "data_process_template/data/layer1_3mer"
outputPath = "predict/layer1"

os.system('python run_enhancer.py '
    '--model_type dna '
    '--tokenizer_name=' + modelPath + '/vocab.txt  '
    '--model_name_or_path ' + modelPath + ' '
   ' --task_name dnaprom '
   ' --do_predict '
  '  --data_dir ' + dataPath + '   '
   ' --max_seq_length 200 '
    '--per_gpu_pred_batch_size=32 '  
    '--output_dir ' + modelPath + '  '
   ' --predict_dir ' + outputPath + '  '
    '--n_process 4'
)

