from torch.utils.data import TensorDataset, DataLoader

def train_rqvae():
    # 1. 配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 256
    EPOCHS = 50
    LR = 1e-3
    
    # 路径
    DATA_PATH = "./processed_data/train_data.pt"
    STATE_CLASSES = "./processed_data/state_classes.npy"
    CITY_CLASSES = "./processed_data/city_classes.npy"
    
    # 2. 加载数据
    print("Loading preprocessed data...")
    data_dict = torch.load(DATA_PATH, weights_only=False)
    features = data_dict['features'] # [N, input_dim]
    state_ids = data_dict['state_ids']
    city_ids = data_dict['city_ids']
    
    # 归一化特征 (重要：便于MSE计算)
    features = F.normalize(features, p=2, dim=1)
    
    dataset = TensorDataset(features, state_ids, city_ids)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    num_states = len(np.load(STATE_CLASSES))
    num_cities = len(np.load(CITY_CLASSES))
    input_dim = features.shape[1]
    
    print(f"Input Dim: {input_dim}, States: {num_states}, Cities: {num_cities}")
    
    # 3. 初始化模型
    model = GeoConstrainedRQVAE(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=4,        # 语义ID长度
        codebook_size=256,   # 每一层的词表大小
        num_states=num_states,
        num_cities=num_cities
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    
    # 4. 训练循环
    model.train()
    for epoch in range(EPOCHS):
        total_loss_avg = 0
        geo_acc_1 = 0 # State Accuracy
        
        # 动态调整 Geo Loss 权重 (Annealing)
        # 初期强约束地理，后期弱化以允许细粒度语义
        lambda_geo = max(0.1, 1.0 - (epoch / (EPOCHS * 0.5)))
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_x, batch_state, batch_city in pbar:
            batch_x = batch_x.to(DEVICE)
            batch_state = batch_state.to(DEVICE)
            batch_city = batch_city.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            out = model(batch_x, state_labels=batch_state, city_labels=batch_city)
            
            # Loss Components
            l_recon = out['recon_loss']
            l_vq = out['vq_loss']
            l_div = out['div_loss']
            l_geo = out['geo_loss']
            
            # Total Loss
            # recon: 1.0, vq: 1.0, div: 0.1, geo: 动态
            loss = l_recon + 1.0 * l_vq + 0.1 * l_div + lambda_geo * l_geo
            
            loss.backward()
            optimizer.step()
            
            total_loss_avg += loss.item()
            
            # 简单的 Accuracy 监控 (仅看 Level 1 是否收敛)
            # 注意：如果只用 California 数据，这个acc始终是 100%
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}", 
                'Recon': f"{l_recon.item():.4f}",
                'Geo': f"{l_geo.item():.4f}"
            })
            
        print(f"Epoch {epoch+1} Finished. Avg Loss: {total_loss_avg / len(dataloader):.4f}")
        
        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"rqvae_epoch_{epoch+1}.pth")

    # 5. 生成所有POI的语义ID (Inference)
    print("Generating Semantic IDs...")
    model.eval()
    all_indices = []
    with torch.no_grad():
        # 不打乱顺序加载
        infer_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        for batch_x, _, _ in infer_loader:
            batch_x = batch_x.to(DEVICE)
            out = model(batch_x)
            # indices: [Batch, 4]
            all_indices.append(out['indices'].cpu())
            
    all_indices = torch.cat(all_indices, dim=0).numpy()
    
    # 保存结果供下一步 (SFT) 使用
    result_df = pd.DataFrame(all_indices, columns=[f'code_{i}' for i in range(4)])
    result_df['gmap_id'] = data_dict['gmap_ids']
    result_df.to_csv("poi_semantic_ids.csv", index=False)
    print("Semantic IDs saved to poi_semantic_ids.csv")

if __name__ == "__main__":
    # 步骤 1: 如果没有数据，先运行 Preprocessor
    # processor = GoogleLocalPreprocessor('path/to/meta-data.json.gz')
    # processor.process()
    
    # 步骤 2: 训练
    train_rqvae()
    # 训练结束后调用评估
    print("\n" + "="*50)
    print("Evaluating Best Model...")

    # 重新加载最佳权重
    model.load_state_dict(torch.load("best_rqvae_model.pth"))

    evaluator = RQVAEEvaluator(
        model=model,
        dataloader=dataloader, # 这里建议传入验证集(Validation Set)，如果没有划分则用训练集
        device=DEVICE,
        output_dir="./evaluation_report"
    )

    metrics = evaluator.run_full_evaluation()

    # 打印最终摘要
    print("\nEvaluation Summary:")
    print(f"Perplexity (L1): {metrics['perplexity']['layer_1']['ppl']:.2f}")
    print(f"State Accuracy:  {metrics['geo_accuracy']['state_acc']*100:.2f}%")
    print(f"Collision Rate:  {metrics['collision_rate']*100:.4f}%")