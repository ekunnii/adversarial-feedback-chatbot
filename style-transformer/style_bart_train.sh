source ~/chatbot_env/bin/activate
pip freeze

cd style-transformer/ && stdbuf -oL python -u main.py \
--data_dir=feedbacks \
--model_type=bart \
--model_name_or_path=bart-large-cnn \
--learning_rate=3e-5 \
--do_lower_case \
--train_batch_size=1 \
--output_dir=feedback_sum \
--do_train \
--num_train_epochs=1
