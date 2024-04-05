from Bio import SeqIO


def extract_and_save_top_percent(fasta_file, output_file, percent=1):
    # 读取原始FASTA文件
    records = list(SeqIO.parse(fasta_file, "fasta"))

    # 计算要提取的记录数
    num_records = len(records)
    top_percent_index = int(num_records * (percent / 100))

    # 提取前10%的记录
    top_records = records[:top_percent_index]

    # 将这些记录保存到新的FASTA文件
    SeqIO.write(top_records, output_file, "fasta")


# 示例用法
input_fasta = './datatest/train.fa'
output_fasta = './datatest/train_top_1_percent.fa'
extract_and_save_top_percent(input_fasta, output_fasta)
