import os

modelPath = "pretrain_model/3-new-12w-0"
step = "2000"

dataPath = "data_process_template/data/layer1_3mer"
outputPath = "result/"

os.system('python run_enhancer.py   '
              ' --model_type dna    '
              ' --tokenizer_name=' + modelPath + '/vocab.txt   '
              ' --model_name_or_path   ' + modelPath + '   '
              ' --task_name dnaprom    '
              ' --do_train    '
              ' --do_eval   '
              ' --data_dir ' + dataPath + '   '
              ' --max_seq_length 200    '
              ' --per_gpu_eval_batch_size=128    '
              ' --per_gpu_train_batch_size=32    '
              ' --learning_rate 4e-4   '
              ' --max_steps ' + step + '  '
              ' --output_dir ' + outputPath + '   '
              ' --evaluate_during_training    '
              ' --logging_steps 100    '
              ' --save_steps 100  '
              ' --early_stop  5     '
              ' --hidden_dropout_prob 0.1     '
              ' --overwrite_output   '
              ' --weight_decay 0.01    '
              ' --n_process 8')

