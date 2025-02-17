# 基于Alpine的轻量级Docker镜像
FROM python:3.9-alpine

# 系统级优化
RUN apk add --no-cache libgomp libstdc++

# 构建阶段
WORKDIR /app
COPY requirements.txt .

# 安装Python依赖（使用清华镜像源加速）
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements.txt --user

# 复制应用程序
COPY . .

# 内存限制配置
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# 启动命令（单工作进程模式）
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5000", "app:server"] 