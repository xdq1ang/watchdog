## 安装MiniConda
略过
## 创建虚拟环境
```
conda create --name your_env_name python=3.8
```

## 删除虚拟环境
```
conda env remove --name your_env_name
```

## 激活虚拟环境
```
conda activate your_env_name
```

## python创建虚拟环境并允许访问主环境中的系统包
venv 提供了一个选项 --system-site-packages，它允许虚拟环境访问主环境中的系统包。请注意，这种方式允许虚拟环境访问主环境中的包，但虚拟环境中的包和主环境中的包是独立的，因此不会自动同步。
```
python3 -m venv --system-site-packages myenv
```