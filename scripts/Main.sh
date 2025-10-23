python -u main.py --data ECL --long_input_len 672  --short_input_len 336 --pred_len 96,192,336,720,1200 --encoder_layers 4,4,4 --decoder_layers 2,2,2 --patch_size 6 --d_model 128 --decoder_IN --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 10  --train --patience 1 --decay 0.5

python -u main.py --data Solar --long_input_len 288  --short_input_len 144 --pred_len 96,192,336,720,1200 --encoder_layers 4,4,4 --decoder_layers 2,2,2 --patch_size 6 --d_model 128 --decoder_IN --learning_rate 0.0001 --dropout 0.1 --batch_size 4 --train_epochs 10 --itr 10  --train --patience 1 --decay 0.5

python -u main.py --data Wind --long_input_len 672  --short_input_len 336 --pred_len 96,192,336,720,1200 --encoder_layers 4,4,4 --decoder_layers 2,2,2 --patch_size 6 --d_model 128 --decoder_IN --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 10 --itr 10  --train --patience 1 --decay 0.5

python -u main.py --data Hydro --long_input_len 672  --short_input_len 336 --pred_len 96,192,336,720 --encoder_layers 4,4,4 --decoder_layers 2,2,2 --patch_size 6 --d_model 128 --decoder_IN --learning_rate 0.0001 --dropout 0.1 --batch_size 16 --train_epochs 10 --itr 10  --train --patience 1 --decay 0.5
