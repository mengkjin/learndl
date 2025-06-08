#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TFT股票预测模型演示脚本 - 最终版本
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model.model_module.tft_model_final import *

def main():
    """TFT模型演示"""
    print("🚀 TFT股票收益率预测模型演示 (最终版本)")
    print("=" * 60)
    
    # 设置参数
    config = {
        'num_stocks': 50,
        'num_days': 200, 
        'seq_len': 20,
        'pred_len': 5,
        'num_industries': 10,
        'hidden_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 1e-3
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 使用设备: {device}")
    
    # 1. 生成数据
    print("\n📊 生成合成股票数据...")
    data = generate_synthetic_data(
        config['num_stocks'], config['num_days'], 
        config['seq_len'], config['pred_len'], config['num_industries']
    )
    
    print(f"✅ 数据形状:")
    print(f"  - 价格: {data['prices'].shape}")
    print(f"  - 成交量: {data['volumes'].shape}")
    print(f"  - 日收益率: {data['returns'].shape}")
    print(f"  - 未来收益率: {data['future_returns'].shape}")
    print(f"  - 行业标签: {data['industries'].shape}")
    
    sequences = create_sequences(data, config['seq_len'], config['pred_len'])
    print(f"✅ 生成 {len(sequences)} 个训练序列")
    
    if len(sequences) == 0:
        print("❌ 错误：没有生成任何训练序列！")
        return
    
    # 检查第一个序列的形状
    first_seq = sequences[0]
    print(f"📋 序列数据形状:")
    print(f"  - 静态特征: {first_seq['static'].shape}")
    print(f"  - 历史特征: {first_seq['historical'].shape}")
    print(f"  - 未来特征: {first_seq['future'].shape}")
    print(f"  - 目标: {first_seq['target'].shape}")
    
    # 2. 数据划分
    split_idx = int(0.8 * len(sequences))
    train_data = sequences[:split_idx]
    val_data = sequences[split_idx:]
    print(f"📈 训练集: {len(train_data)} 个序列")
    print(f"📉 验证集: {len(val_data)} 个序列")
    
    # 3. 创建模型
    print("\n🧠 创建TFT模型...")
    model = TemporalFusionTransformer(
        static_dim=config['num_industries'],
        known_dynamic_dim=2,  # 价格 + 成交量
        unknown_dynamic_dim=1,  # 历史收益率
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len'],
        quantiles=[0.1, 0.5, 0.9]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔢 模型参数数量: {total_params:,}")
    
    # 4. 训练模型
    print(f"\n🏋️ 开始训练模型 ({config['num_epochs']} epochs)...")
    train_losses, val_losses = train_model(
        model, train_data, val_data,
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        device=device
    )
    
    # 5. 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(val_losses, label='验证损失', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('TFT模型训练曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_curves_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. 测试推理
    print("\n🔮 测试模型推理...")
    test_samples = val_data[:5]
    
    print("\n📋 预测结果示例:")
    for i, sample in enumerate(test_samples):
        pred_result = predict(
            model, sample['static'], sample['historical'], 
            sample['future'], device
        )
        
        predictions = pred_result['predictions']
        target = sample['target']
        
        print(f"\n样本 {i+1}:")
        print("  预测 vs 真实 (10%, 50%, 90%分位数):")
        for day in range(config['pred_len']):
            pred_day = predictions[day]
            target_day = target[day]
            print(f"    第{day+1}天: [{pred_day[0]:.4f}, {pred_day[1]:.4f}, {pred_day[2]:.4f}] vs {target_day:.4f}")
    
    # 7. 评估模型
    print("\n📊 模型整体评估:")
    all_predictions = []
    all_targets = []
    
    for sample in val_data[:20]:  # 评估前20个样本
        pred_result = predict(
            model, sample['static'], sample['historical'], 
            sample['future'], device
        )
        all_predictions.append(pred_result['predictions'])
        all_targets.append(sample['target'])
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # 展平用于评估
    pred_flat = all_predictions.reshape(-1, 3)
    target_flat = all_targets.reshape(-1)
    
    eval_results = evaluate_predictions(pred_flat, target_flat)
    
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # 8. 可视化预测结果
    sample_idx = 0
    sample_pred = all_predictions[sample_idx]
    sample_target = all_targets[sample_idx]
    
    plt.figure(figsize=(12, 8))
    days = range(1, config['pred_len'] + 1)
    
    plt.fill_between(days, sample_pred[:, 0], sample_pred[:, 2], 
                     alpha=0.3, color='blue', label='80%置信区间')
    plt.plot(days, sample_pred[:, 1], 'b-', linewidth=2, label='中位数预测')
    plt.plot(days, sample_target, 'ro-', linewidth=2, label='真实值')
    
    plt.xlabel('预测天数')
    plt.ylabel('收益率')
    plt.title('TFT股票收益率预测结果示例')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('prediction_example_final.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ TFT模型演示完成！")
    print("📁 生成的文件:")
    print("  - training_curves_final.png: 训练曲线")
    print("  - prediction_example_final.png: 预测结果示例")

if __name__ == "__main__":
    main() 