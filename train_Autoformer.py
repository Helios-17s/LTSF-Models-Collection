
import os
import subprocess
import sys

# ==========================================================
# 【核心参数配置区域】 - 您只需修改此处即可适配新数据集
# ==========================================================

# 1. 数据路径配置
DATA_ROOT = "/content/"            # 数据文件所在的根目录
DATA_FILE = "dengue_processed.csv" # 数据文件名 (CSV格式)
MODEL_ID = "my_timeseries_task"     # 实验任务名称，用于保存模型和日志

# 2. 时间序列维度配置
SEQ_LEN = 96      # 输入序列长度 (看过去多少个时间步)
LABEL_LEN = 48    # 标签长度 (通常为预测长度的一半，用于解码器引导)
PRED_LEN = 96     # 预测未来长度 (预测未来多少个时间步)

# 3. 特征维度配置
ENC_IN = 52       # 编码器输入维度 (特征列的总数)
DEC_IN = 52       # 解码器输入维度 (通常与ENC_IN一致)
C_OUT = 52        # 输出维度 (预测目标的列数)

# 4. 训练超参数
EPOCHS = 10       # 训练总轮数
BATCH_SIZE = 8    # 批大小 (CPU环境下建议设置在4-16之间)
LEARNING_RATE = 0.0001 # 学习率

# 5. 硬件配置
USE_GPU = "False" # 是否使用GPU (Colab无显存时请保持False)
# ==========================================================

def apply_patches():
    """自动应用针对LTSF框架的CPU兼容性补丁"""
    # 修复 AutoCorrelation 层的硬编码 .cuda() 问题
    corr_path = 'layers/AutoCorrelation.py'
    if os.path.exists(corr_path):
        with open(corr_path, 'r') as f: content = f.read()
        if '.cuda()' in content:
            with open(corr_path, 'w') as f: f.write(content.replace('.cuda()', '.to(values.device)'))
    
    # 修复实验类的 GPU 检测逻辑
    exp_path = 'exp/exp_main.py'
    if os.path.exists(exp_path):
        with open(exp_path, 'r') as f: content = f.read()
        if 'if self.args.use_gpu:' in content:
            with open(exp_path, 'w') as f: f.write(content.replace('if self.args.use_gpu:', 'import torch\n        if self.args.use_gpu and torch.cuda.is_available():'))

def run_training():
    apply_patches()
    
    cmd = [
        "python", "-u", "run_longExp.py",
        "--is_training", "1",
        "--root_path", DATA_ROOT,
        "--data_path", DATA_FILE,
        "--model_id", MODEL_ID,
        "--model", "Autoformer",
        "--data", "custom",
        "--features", "M", # M:多变量预测, S:单变量预测
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
    
    print(f"--- 启动 Autoformer 训练任务: {MODEL_ID} ---")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        sys.stdout.write(line)
    process.wait()

if __name__ == "__main__":
    run_training()
