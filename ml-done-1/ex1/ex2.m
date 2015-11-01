clear; close all; clc

fprintf('Testing the plot \n');
fprintf('Plotting the data \n');

%loading the data from the file
data = load('ex1data1.txt');
X = data(:,1);
Y = data(:,2);
fprintf('Number of training examples is %d \n',length(Y));
%plot the data

plot(X,Y,'rx');
xlabel('Poppulation in 10,000s');
ylabel('Price in 10,000$');
