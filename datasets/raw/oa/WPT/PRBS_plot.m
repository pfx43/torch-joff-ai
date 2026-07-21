% 读取mat文件并绘图
load('E:\状态配准神经网络观测器\3WPT应用\PRBS1024\PRBS_1024_u');
load('E:\状态配准神经网络观测器\3WPT应用\PRBS1024\PRBS_1024_y');
t_k = yk(1,:);
u_k = uk(2,:);
y_k = yk(2,:);


% 计算原始数据列的功率
P = mean(y_k.^2);
% 计算所需的噪声功率
% SNR = 60; % dB
N0 = 0.0001;
SNR = 10*log10(P/N0)
% 生成相同长度的白噪声序列
noise = sqrt(N0) * randn(size(y_k));
% 将噪声添加到原始数据中
noisy_y_k = y_k + noise;


figure
plot(uk(1,:),u_k)
figure
plot(yk(1,:),y_k)
xlim([1.5 1.6])
figure
plot(t_k,noisy_y_k);
xlim([1.5 1.6])


data = [u_k' noisy_y_k'];
filename = 'E:\状态配准神经网络观测器\3WPT应用\PRBS1024\data1024.xlsx';
sheetname = 'ukyk';
% xlswrite(filename, data, sheetname);





