clc;
clear;
data=textread('C:\Users\12184\Desktop\10um.txt','%f');
n=2:2:size(data);
y=data(n)';
m=1:2:size(data);
t=data(m)';
[i,j]=size(m);[ii,jj]=size(m);
fs=j/t(1,jj);
Ts=1/fs;
N=j;
delta_f=1*fs/N;
figure(1);
subplot(3,1,1)
plot(t,y);
Y=fftshift(abs(fft(y)));
f=(-N/2:N/2-1)*delta_f;
subplot(3,1,2)
plot(f,Y);
wp=2/(fs/2); %Filters out noise smaller than that frequency
ws=10/(fs/2); %Less than but close to the required frequency
alpha_p=0.005; %Attenuation intensity close to the required frequency
alpha_s=25;%Noise attenuation intensity
[N4,wn]=buttord(wp,ws,alpha_p,alpha_s);
[b,a]=butter(N4,wn,'high');
filter_bs_s = filter(b,a,y);
Y_bs_s = fftshift(abs(fft(filter_bs_s)))/N;
figure(2);
subplot(3,1,1);
plot(t,filter_bs_s);
grid on;
title('xxx');
subplot(3,1,2);
plot(f,Y_bs_s);
title('xxx');

for i=1:jj
if t(1,i)>0.01
    p=i;
    break;
end
end
tt=zeros(1,jj-p+1);filter_bs_ss=zeros(1,jj-p+1);
for i=p:jj 
tt(1,i-p+1)=t(1,i);
end
for j=p:jj
    filter_bs_ss(1,j-p+1)=filter_bs_s(1,j);
end

figure(3)
subplot(2,2,1)
plot(t,filter_bs_s);
[sst,f]=wsst(filter_bs_s,jj/t(1,jj-p+1));
subplot(2,2,2)
pcolor(t,f,abs(sst));shading flat                                                                                                                                                                                                                                 
subplot(2,2,3)
plot(tt,filter_bs_ss);
[sst,f]=wsst(filter_bs_ss,jj/tt(1,jj-p+1));
subplot(2,2,4)
pcolor(tt,f,abs(sst));shading flat

ttt=zeros(1,jj-p+1);
for i=1:jj-p+1
    ttt(1,i)=tt(1,i)-tt(1,1);
end
figure(4)
plot(ttt,filter_bs_ss);

figure(5)
pcolor(ttt,f,abs(sst));shading flat                                                                                                                                                                                                                                 ;
colorbar;

set(gca,'fontsize',32,'fontname','Times new roman','fontweight','bold');
set(gca,'YDir','normal')
xlabel('Time/s','Fontweight','bold');

axis([0 100 0 200]);
set(gca,'XTick',0:20:100);
set(gca,'YTick',0:30:200);
title('Wavelet transform','fontweight','bold');


width=500;
height=400;
left=200;
bottem=100;
set(gcf,'position',[left,bottem,width,height])

jjj(1,:)=ttt(1,:);
jjj(2,:)=filter_bs_ss(1,:);
jjj=jjj';
xlswrite('C:\Users\12184\Desktop\10um.xlsx',jjj,1);
