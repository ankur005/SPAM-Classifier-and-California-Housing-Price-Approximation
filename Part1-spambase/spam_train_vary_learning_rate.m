P_train = P_train_std; 
Val.P = Val_std.P; %%% Use this line if you use STD preprocessing on the data. IMPORTANT: Run preprocess.m first 
hiddenLayerSize = [10];
net = fitnet(hiddenLayerSize);
net.trainFcn = 'traingd'; 
net.layers{2}.transferFcn = 'tansig'; %Hidden layer function
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 30/100;
net.divideParam.testRatio = 0/100;
net.trainParam.epochs =1000;
net.trainParam.max_fail = 25;
lr = transpose([0.001,0.005,0.01,0.05,0.1,0.5,1]);
missclassificationRate = zeros(7,1);
disp(size(lr));
for i = 1:size(lr)
    net.trainParam.lr = lr(i);
    [net tr] = train(net,P_train,T_train);
    [fields N] = size(T_test);
    neuralnetscore = sign(net(Val.P));
    missclassificationRate(i) = sum(0.5*abs(T_test - neuralnetscore))/N;
    savePerformancePlot(['Performance_',strrep(num2str(lr(i)),'.','_')],tr);
    saveErrorHistogram(['ErrorHist_',strrep(num2str(lr(i)),'.','_')],gsubtract(T_train,net(P_train)));
    saveTrainStatePlot(['TrainState_',strrep(num2str(lr(i)),'.','_')],tr);
    saveRegressionPlot(['Reg_',strrep(num2str(lr(i)),'.','_')],T_train,net(P_train));
end

%saveMissclassificationPlot('lr_missclassificationRate',lr,missclassificationRate);

function savePerformancePlot(figureName,tr)
    fileName = ['Figures\VaryAlpha\Performance\',figureName];
    h = figure;
    plotperform(tr);
    saveas(h,[fileName,'.jpg']);
end

function saveErrorHistogram(figureName,graphInput)
    fileName = ['Figures\VaryAlpha\ErrorHistogram\',figureName];
    h = figure;
    ploterrhist(graphInput);
    saveas(h,[fileName,'.jpg']);
end

function saveRegressionPlot(figureName,T_train, output)
    fileName = ['Figures\VaryAlpha\Regression\',figureName];
    h = figure;
    plotregression(T_train,output);
    saveas(h,[fileName,'.jpg']);
end

function saveTrainStatePlot(figureName,tr)
    fileName = ['Figures\VaryAlpha\TrainState\',figureName];
    h = figure;
    plottrainstate(tr);
    saveas(h,[fileName,'.jpg']);
end

function saveMissclassificationPlot(figureName,lr,misclassificationRate)
    fileName = ['Figures\VaryAlpha\MissclassificationRate\',figureName];
    h = figure;
    plot(lr,misclassificationRate);
    saveas(h,[fileName,'.jpg']);
end