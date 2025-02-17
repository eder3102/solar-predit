# 系统监控方案

## 监控架构

### 1. 监控组件
- Prometheus: 指标收集
- Grafana: 可视化面板
- AlertManager: 告警管理
- Node Exporter: 主机监控
- cAdvisor: 容器监控

### 2. 监控指标

#### 2.1 系统指标
| 指标 | 说明 | 阈值 | 告警级别 |
|------|------|------|----------|
| CPU使用率 | 系统CPU使用率 | >80% | 警告 |
| 内存使用 | 系统内存使用 | >3.5GB | 严重 |
| 磁盘使用 | 磁盘空间使用率 | >85% | 警告 |
| 网络流量 | 网络带宽使用率 | >80% | 警告 |

#### 2.2 应用指标
| 指标 | 说明 | 阈值 | 告警级别 |
|------|------|------|----------|
| API延迟 | 接口响应时间 | >500ms | 警告 |
| 错误率 | 接口错误率 | >1% | 严重 |
| QPS | 每秒请求数 | >100 | 警告 |
| 并发数 | 同时在线请求数 | >50 | 警告 |

#### 2.3 模型指标
| 指标 | 说明 | 阈值 | 告警级别 |
|------|------|------|----------|
| 预测误差 | MAE指标 | >0.15 | 严重 |
| 预测延迟 | 单次预测用时 | >100ms | 警告 |
| 模型内存 | 模型内存占用 | >1GB | 警告 |
| 预测失败率 | 预测失败比例 | >0.1% | 严重 |

## 监控配置

### 1. Prometheus配置
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'solar-predict'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scheme: 'http'
```

### 2. Grafana面板
```json
{
  "dashboard": {
    "panels": [
      {
        "title": "系统资源监控",
        "type": "graph",
        "targets": [
          {
            "expr": "node_cpu_usage",
            "legendFormat": "CPU使用率"
          },
          {
            "expr": "node_memory_usage",
            "legendFormat": "内存使用"
          }
        ]
      },
      {
        "title": "API性能监控",
        "type": "graph",
        "targets": [
          {
            "expr": "http_request_duration_seconds",
            "legendFormat": "接口延迟"
          }
        ]
      }
    ]
  }
}
```

### 3. 告警规则
```yaml
# alerting_rules.yml
groups:
  - name: solar_predict_alerts
    rules:
      - alert: HighMemoryUsage
        expr: node_memory_usage_bytes > 3.5e9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "内存使用超限"
          description: "当前内存使用: {{ $value | humanize }}GB"

      - alert: HighPredictionError
        expr: prediction_error_mae > 0.15
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "预测误差过大"
          description: "当前MAE: {{ $value }}"
```

## 监控指标采集

### 1. 系统指标采集
```python
# metrics.py
from prometheus_client import Gauge, Counter, Histogram

# 系统指标
cpu_usage = Gauge('cpu_usage_percent', 'CPU使用率')
memory_usage = Gauge('memory_usage_bytes', '内存使用量')
disk_usage = Gauge('disk_usage_percent', '磁盘使用率')

# 应用指标
request_latency = Histogram('http_request_duration_seconds', '接口延迟')
request_count = Counter('http_requests_total', '请求总数')
error_count = Counter('http_errors_total', '错误总数')

# 模型指标
prediction_error = Gauge('prediction_error_mae', '预测MAE')
prediction_latency = Histogram('prediction_duration_seconds', '预测用时')
model_memory = Gauge('model_memory_bytes', '模型内存')
```

### 2. 指标更新
```python
# app.py
@app.before_request
def before_request():
    request.start_time = time.time()

@app.after_request
def after_request(response):
    # 更新请求指标
    latency = time.time() - request.start_time
    request_latency.observe(latency)
    request_count.inc()
    
    if response.status_code >= 400:
        error_count.inc()
    
    return response

def update_system_metrics():
    """更新系统指标"""
    while True:
        cpu_usage.set(psutil.cpu_percent())
        memory_usage.set(psutil.virtual_memory().used)
        disk_usage.set(psutil.disk_usage('/').percent)
        time.sleep(15)
```

## 监控面板

### 1. 系统概览
- CPU/内存/磁盘使用率
- 网络流量监控
- 系统负载趋势

### 2. API监控
- 接口调用量
- 响应时间分布
- 错误率统计
- QPS趋势

### 3. 模型监控
- 预测准确率
- 预测延迟
- 模型资源使用
- 异常预测统计

## 告警处理

### 1. 告警级别
| 级别 | 响应时间 | 处理方式 |
|------|----------|----------|
| 严重 | 15分钟内 | 电话+短信+邮件 |
| 警告 | 30分钟内 | 短信+邮件 |
| 提示 | 2小时内 | 邮件 |

### 2. 告警升级
1. L1: 运维工程师
   - 系统资源告警
   - 基础服务告警
   
2. L2: 系统工程师
   - 应用服务告警
   - 性能问题告警
   
3. L3: 开发团队
   - 模型异常告警
   - 核心功能告警

### 3. 告警抑制
```yaml
# alertmanager.yml
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['instance', 'job']
```

## 监控运维

### 1. 日常维护
- 监控服务健康检查
- 监控数据备份
- 告警规则更新
- 面板配置优化

### 2. 容量规划
- 监控数据存储容量
- 时序数据压缩配置
- 历史数据清理策略
- 备份策略优化

### 3. 故障处理
- 监控服务异常处理
- 数据采集问题排查
- 告警风暴处理
- 监控恢复流程 