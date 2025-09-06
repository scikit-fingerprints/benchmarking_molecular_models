log_dir="reproduce_posebuster"
mkdir -p log/$log_dir
init_model="model/helixdock.pdparams"
dataset_config="configs/dataset_configs/poesbusters.json"
model_config="configs/model_configs/helixdock_model.json"
train_config="configs/train_configs/lr8e-4_ema.json"
encoder_config="configs/model_configs/helixdock_encoder.json"

python paddle.distributed.launch --log_dir log/$log_dir evalute.py \
--batch_size 2 \
--init_model $init_model \
--distributed \
--encoder_config $encoder_config \
--dataset_config $dataset_config \
--train_config $train_config \
--model_config $model_config \
--log_dir log/$log_dir
