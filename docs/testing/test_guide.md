# 光伏发电预测系统测试文档

## 测试环境

### 1. 测试环境配置
- 操作系统: Ubuntu 20.04 LTS
- Python: 3.8.10
- 内存限制: 4GB
- CPU限制: 2核心

### 2. 测试数据集
```python
# 数据集结构
data/
├── train/
│   ├── 2023_01_06.csv  # 晴天数据
│   ├── 2023_02_15.csv  # 阴天数据
│   └── 2023_03_22.csv  # 雨天数据
├── test/
│   ├── normal/         # 正常场景
│   ├── extreme/        # 极端天气
│   └── missing/        # 缺失数据
└── validation/
    ├── 2024_01.csv     # 验证集1
    └── 2024_02.csv     # 验证集2
```

## 测试规范

### 1. 单元测试
```python
# tests/test_models.py
def test_filter_net():
    """测试FilterNet模型"""
    model = FilterNet()
    x = torch.randn(16, 91, 96)
    y = model(x)
    
    assert y.shape == (16, 96)
    assert not torch.isnan(y).any()
    assert torch.all(y >= 0)

# tests/test_data.py
def test_data_loader():
    """测试数据加载器"""
    loader = DataLoader('data/train')
    batch = next(iter(loader))
    
    assert batch['features'].shape == (16, 91, 96)
    assert batch['labels'].shape == (16, 96)
    assert not torch.isnan(batch['features']).any()
```

### 2. 集成测试
```python
# tests/test_integration.py
def test_prediction_pipeline():
    """测试完整预测流程"""
    # 1. 加载数据
    data = load_test_data()
    
    # 2. 特征处理
    features = preprocess_features(data)
    
    # 3. 模型预测
    predictions = model.predict(features)
    
    # 4. 验证结果
    assert predictions.shape == (96,)
    assert 0 <= predictions.mean() <= 1
    assert calculate_metrics(predictions, data['labels'])['mae'] < 0.15
```

### 3. 性能测试
```python
# tests/test_performance.py
def test_memory_usage():
    """测试内存使用"""
    process = psutil.Process()
    
    # 1. 基准内存
    base_memory = process.memory_info().rss
    
    # 2. 加载模型
    model = load_model()
    model_memory = process.memory_info().rss
    
    # 3. 批量预测
    for _ in range(100):
        model.predict(get_test_batch())
    peak_memory = process.memory_info().rss
    
    assert (peak_memory - base_memory) / 1024**2 < 3500  # 小于3.5GB

def test_prediction_latency():
    """测试预测延迟"""
    model = load_model()
    
    latencies = []
    for _ in range(1000):
        start = time.time()
        model.predict(get_test_batch())
        latencies.append(time.time() - start)
    
    assert np.mean(latencies) < 0.5  # 平均延迟<500ms
    assert np.percentile(latencies, 95) < 0.8  # P95<800ms
```

## 测试用例

### 1. 功能测试用例

#### 1.1 数据处理测试
| 用例ID | 测试场景 | 输入数据 | 预期结果 | 验证方法 |
|--------|----------|----------|-----------|----------|
| DP001  | 正常数据 | 完整的96点数据 | 特征正确提取 | 检查特征维度和范围 |
| DP002  | 缺失数据 | 包含null的数据 | 正确插值处理 | 验证插值结果合理性 |
| DP003  | 异常数据 | 包含离群值 | 异常值被处理 | 检查处理后的数据分布 |

#### 1.2 模型预测测试
| 用例ID | 测试场景 | 输入条件 | 预期结果 | 验证方法 |
|--------|----------|----------|-----------|----------|
| MP001  | 晴天预测 | 晴天数据 | MAE < 0.1 | 计算预测误差 |
| MP002  | 阴天预测 | 阴天数据 | MAE < 0.15 | 计算预测误差 |
| MP003  | 雨天预测 | 雨天数据 | MAE < 0.2 | 计算预测误差 |

### 2. 性能测试用例

#### 2.1 负载测试
| 用例ID | 测试场景 | 测试条件 | 成功标准 | 监控指标 |
|--------|----------|----------|-----------|----------|
| PT001  | 单次预测 | 96点预测 | 延迟<500ms | API响应时间 |
| PT002  | 并发预测 | 10并发   | QPS>20 | 系统吞吐量 |
| PT003  | 持续预测 | 24小时运行 | 内存<3.5GB | 资源使用率 |

#### 2.2 异常恢复测试
| 用例ID | 测试场景 | 操作步骤 | 预期结果 | 验证方法 |
|--------|----------|----------|-----------|----------|
| RT001  | 服务重启 | 重启服务 | 自动恢复 | 检查服务状态 |
| RT002  | 模型切换 | 切换版本 | 正常预测 | 验证预测结果 |
| RT003  | 负载过高 | 模拟高负载 | 服务降级 | 监控系统指标 |

## 测试执行

### 1. 自动化测试
```bash
# 运行所有测试
pytest tests/ -v

# 运行性能测试
pytest tests/test_performance.py -v

# 运行并生成覆盖率报告
pytest --cov=src tests/
```

### 2. 测试报告
```python
# 生成测试报告
def generate_test_report():
    results = {
        'unit_tests': run_unit_tests(),
        'integration_tests': run_integration_tests(),
        'performance_tests': run_performance_tests()
    }
    
    report = TestReport(results)
    report.save('test_report.html')
```

### 3. 测试监控
```python
# 监控测试指标
def monitor_test_metrics():
    metrics = {
        'test_coverage': get_coverage(),
        'test_success_rate': get_success_rate(),
        'performance_metrics': get_performance_metrics()
    }
    
    alert_if_needed(metrics)
    update_dashboard(metrics)
```

## 回归测试

### 1. 回归测试集
```python
# tests/regression/
├── known_issues/      # 已知问题测试
├── fixed_bugs/        # 修复验证测试
└── performance/       # 性能回归测试
```

### 2. 回归测试流程
1. 代码提交触发测试
2. 运行回归测试集
3. 对比历史结果
4. 生成回归报告
5. 通知相关人员

### 3. 回归测试标准
| 类型 | 指标 | 标准 |
|------|------|------|
| 功能 | 测试通过率 | 100% |
| 性能 | 性能劣化 | <5% |
| 内存 | 内存增长 | <100MB |

## 测试工具

### 1. 测试框架
- pytest: 单元测试
- locust: 性能测试
- coverage: 覆盖率分析

### 2. 监控工具
- Prometheus: 指标收集
- Grafana: 可视化面板
- AlertManager: 告警管理

### 3. 辅助工具
```python
# tools/test_data_generator.py
def generate_test_data():
    """生成测试数据"""
    pass

# tools/performance_analyzer.py
def analyze_performance():
    """分析性能指标"""
    pass
```

## 持续测试

### 1. CI/CD集成
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Tests
        run: |
          pip install -r requirements.txt
          pytest tests/
```

### 2. 定时测试
```bash
# crontab配置
0 2 * * * cd /app && pytest tests/regression/
0 0 * * 0 cd /app && pytest tests/performance/
```

### 3. 测试结果通知
```python
# utils/notification.py
def notify_test_results(results):
    """发送测试结果通知"""
    if results['failed']:
        send_alert('测试失败', results)
    
    update_dashboard(results)
    archive_results(results)
``` 