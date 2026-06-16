import subprocess

# 指定 pdflatex 的路径
pdflatex_path = r'E:/latex/texlive/2024/bin/windows/pdflatex.exe'

# 运行 pdflatex 命令
subprocess.run([pdflatex_path, 'E:/UNet_Demo/UNet1/nets/PlotNeuralNet/examples/VGG16/vgg16.tex'], check=True)
