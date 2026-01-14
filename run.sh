#conda activate speech_conversation

#CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 train.py \
#    --config_model configs/ConversationV1 \

#CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 train_lora.py \
#    --config_model configs/ConversationV1 \

#CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 train_qwenllm.py \
#    --config_model configs/ConversationV2 \

#CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=29501 train_qwenllm.py \
#    --config_model configs/Conversation_Qwen0.5B \

#CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29503 train_s2s.py \
#    --config_model configs/Conversation_S2S \

#CUDA_VISIBLE_DEVICES=2 python train_s2s.py \
#    --config_model configs/Conversation_S2S \

##################################################
#CUDA_VISIBLE_DEVICES=2 python test.py
#CUDA_VISIBLE_DEVICES=2 python test_dialogue.py
CUDA_VISIBLE_DEVICES=2 python test_dialogue_inthewild.py
#CUDA_VISIBLE_DEVICES=2 python test_dialogue_S2S.py