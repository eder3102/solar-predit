# 光伏发电预测系统部署文档

## 环境要求

### 1. 硬件要求
- CPU: 2核心
- 内存: 4GB
- 磁盘: 20GB以上
- 网络: 100Mbps以上

### 2. 软件要求
- 操作系统: Ubuntu 20.04 LTS
- Python: 3.8+
- Docker: 20.10+
- NVIDIA Driver: 不需要

## 部署步骤

### 1. 基础环境配置
```bash
# 更新系统
sudo apt-get update
sudo apt-get upgrade -y

# 安装依赖
sudo apt-get install -y \
    python3.8 \
    python3.8-venv \
    python3-pip \
    docker.io \
    git

# 配置Docker
sudo systemctl start docker
sudo systemctl enable docker
```

### 2. 项目部署

#### 2.1 获取代码
```bash
# 克隆项目
git clone https://github.com/your-org/solar-predict.git
cd solar-predict

# 创建虚拟环境
python3.8 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 2.2 Docker部署
```bash
# 构建镜像
docker build -t solar-predict:v1.0.0 .

# 运行容器
docker run -d \
    --name solar-predict \
    -p 8000:8000 \
    -v /data:/app/data \
    --restart always \
    solar-predict:v1.0.0
```

### 3. 配置说明

#### 3.1 环境变量
```bash
# .env文件配置
MODEL_PATH=/app/models
DATA_PATH=/app/data
LOG_LEVEL=INFO
MAX_WORKERS=2
BATCH_SIZE=16
```

#### 3.2 内存优化
```bash
# /etc/sysctl.conf
vm.swappiness=10
vm.vfs_cache_pressure=50

# 应用配置
sudo sysctl -p
```

#### 3.3 日志配置
```yaml
# config/logging.yaml
version: 1
formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
handlers:
  file:
    class: logging.handlers.RotatingFileHandler
    filename: /app/logs/app.log
    maxBytes: 10485760
    backupCount: 5
```

## 系统维护

### 1. 日常维护

#### 1.1 日志检查
```bash
# 检查应用日志
tail -f /app/logs/app.log

# 检查系统日志
journalctl -u solar-predict -f
```

#### 1.2 性能监控
```bash
# 监控系统资源
docker stats solar-predict

# 检查API延迟
curl http://localhost:8000/api/v1/monitor/performance
```

#### 1.3 数据备份
```bash
# 备份模型文件
tar -czf models_$(date +%Y%m%d).tar.gz /app/models/

# 备份配置文件
cp -r /app/config/ /backup/config_$(date +%Y%m%d)/
```

### 2. 故障处理

#### 2.1 常见问题
1. 内存溢出
```bash
# 检查内存使用
free -h

# 清理缓存
echo 3 > /proc/sys/vm/drop_caches
```

2. 服务无响应
```bash
# 重启服务
docker restart solar-predict

# 检查日志
docker logs -f solar-predict
```

3. 预测异常
```bash
# 检查模型状态
curl http://localhost:8000/api/v1/model/status

# 回滚模型版本
./scripts/rollback.sh v1.0.0
```

### 3. 升级维护

#### 3.1 版本升级
```bash
# 停止服务
docker stop solar-predict

# 备份数据
./scripts/backup.sh

# 更新代码
git pull origin main

# 构建新镜像
docker build -t solar-predict:v1.0.1 .

# 启动新版本
docker run -d \
    --name solar-predict-new \
    -p 8000:8000 \
    -v /data:/app/data \
    solar-predict:v1.0.1

# 验证新版本
curl http://localhost:8000/api/v1/monitor/health

# 切换版本
docker rm -f solar-predict
docker rename solar-predict-new solar-predict
```

#### 3.2 数据迁移
```bash
# 导出数据
./scripts/export_data.sh

# 迁移数据
./scripts/migrate_data.sh

# 验证数据
./scripts/verify_data.sh
```

## 监控告警

### 1. 监控指标
| 指标 | 阈值 | 告警级别 |
|------|------|----------|
| CPU使用率 | >80% | 警告 |
| 内存使用 | >3.5GB | 严重 |
| API延迟 | >500ms | 警告 |
| 预测误差 | >15% | 严重 |

### 2. 告警配置
```yaml
# config/alerting.yaml
rules:
  - name: high_memory_usage
    condition: memory_usage > 3.5GB
    duration: 5m
    severity: critical
    channels:
      - email
      - sms
```

### 3. 告警处理
1. 收到告警后，按以下步骤处理：
   - 检查监控面板
   - 分析日志信息
   - 执行相应处理脚本
   - 记录处理过程
   - 更新runbook

2. 告警升级流程：
   - L1: 运维工程师
   - L2: 系统工程师
   - L3: 开发团队

## 安全配置

### 1. 网络安全
```bash
# 配置防火墙
ufw allow 8000/tcp
ufw enable

# 配置SSL
certbot --nginx -d api.example.com
```

### 2. 访问控制
```bash
# 添加API密钥
echo "API_KEY=your-secret-key" >> .env

# 配置认证
curl -H "Authorization: Bearer your-secret-key" \
     http://localhost:8000/api/v1/predict
```

### 3. 日志审计
```bash
# 启用审计日志
auditctl -w /app/models/ -p wa -k model_changes
auditctl -w /app/config/ -p wa -k config_changes

# 查看审计日志
ausearch -k model_changes
```

## 性能优化

### 1. 系统优化
```bash
# 优化内核参数
cat >> /etc/sysctl.conf << EOF
net.core.somaxconn = 1024
net.ipv4.tcp_max_syn_backlog = 1024
net.ipv4.tcp_fin_timeout = 30
EOF

# 优化文件描述符
ulimit -n 65535
```

### 2. 应用优化
```python
# config/gunicorn.py
workers = 2
worker_class = 'uvicorn.workers.UvicornWorker'
keepalive = 65
timeout = 120
max_requests = 1000
max_requests_jitter = 50
```

## 备份策略

### 1. 备份内容
- 模型文件
- 配置文件
- 历史数据
- 日志文件

### 2. 备份计划
| 内容 | 周期 | 保留时间 |
|------|------|----------|
| 模型 | 每日 | 7天 |
| 配置 | 每周 | 30天 |
| 数据 | 每日 | 90天 |
| 日志 | 每周 | 30天 |

### 3. 恢复流程
```bash
# 恢复模型
./scripts/restore_model.sh backup_20240214.tar.gz

# 恢复配置
./scripts/restore_config.sh config_20240214.tar.gz

# 验证恢复
./scripts/verify_restore.sh
``` 