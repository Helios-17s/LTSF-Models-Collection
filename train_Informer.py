
import os
import subprocess
import sys

# ==========================================================
# 【核心参数配置区域】 - 您只需修改此处即可适配新数据集
# ==========================================================

# 1. 数据路径配置
DATA_ROOT = "/content/"            # 数据文件所在的根目录
DATA_FILE = "dengue_processed.csv" # 数据文件名 (CSV格式)
MODEL_ID = "informer_full_train"   # 实验任务名称，用于保存模型和日志

# 2. 时间序列维度配置
SEQ_LEN = 96      # 输入序列长度 (观察过去 96 个时间步)
LABEL_LEN = 48    # 标签长度 (用于解码器引导的已知序列长度)
PRED_LEN = 96     # 预测未来长度 (预测未来 96 个时间步)

# 3. 特征维度配置
ENC_IN = 52       # 编码器输入维度 (特征列的总数，登革热数据为 52)
DEC_IN = 52       # 解码器输入维度 (通常与 ENC_IN 一致)
C_OUT = 52        # 输出维度 (预测目标的列数)

# 4. 训练超参数
EPOCHS = 10       # 训练总轮数 (已修改为 10)
BATCH_SIZE = 8    # 批大小 (CPU 环境建议 4-16)
LEARNING_RATE = 0.0001 # 学习率

# 5. 硬件配置
USE_GPU = "False" # 是否使用 GPU (Colab 没显存时保持 False)
# ==========================================================

def run_training():
    # 构建 Informer 运行指令
    cmd = [
        "python", "-u", "run_longExp.py",
        "--is_training", "1",
        "--root_path", DATA_ROOT,
        "--data_path", DATA_FILE,
        "--model_id", MODEL_ID,
        "--model", "Informer",
        "--data", "custom",
        "--features", "M",      # M: 多变量预测
        "--seq_len", str(SEQ_LEN),
        "--label_len", str(LABEL_LEN),
        "--pred_len", str(PRED_LEN),
        "--enc_in", str(ENC_IN),
        "--dec_in", str(DEC_IN),
        "--c_out", str(C_OUT),
        "--train_epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--learning_rate", str(LEARNING_RATE),
        "--use_gpu", USE_GPU,
        "--itr", "1"
    ]

    print(f"--- 启动 Informer 训练任务: {MODEL_ID} (共 {EPOCHS} 轮) ---")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # 实时捕获日志并输出，确保能看到每轮的 MAE/MSE
    for line in process.stdout:
        sys.stdout.write(line)
    
    process.wait()
    print("\n训练任务执行完毕。")

if __name__ == "__main__":
    run_training()
